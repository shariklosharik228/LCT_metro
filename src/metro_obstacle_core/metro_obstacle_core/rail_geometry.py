"""Rail-pair continuation and normal coordinates; no straight-axis clipping."""
import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline


class PathModel:
    valid = True

    def __init__(self, observations, center_nodes=None, z_nodes=None):
        values = np.asarray(observations)
        self.x_nodes = values[:, 0]
        self.path_center = (UnivariateSpline(self.x_nodes, values[:,1],
                              s=len(values)*.08**2)(self.x_nodes)
                            if center_nodes is None else np.asarray(center_nodes,dtype=float).copy())
        if self.path_center.shape != self.x_nodes.shape or not np.isfinite(self.path_center).all():
            raise ValueError('Invalid centreline nodes')
        # A dense near section is stronger evidence than isolated far returns.
        # Do not bend the vertical datum down onto distant ballast/noise.
        weights = np.sqrt(values[:,4]) if values.shape[1]>4 else np.ones(len(values))
        z_coeff = np.polyfit(self.x_nodes, values[:,2], 1, w=weights)
        if z_nodes is None:
            self.z_nodes = np.polyval(z_coeff, self.x_nodes)
        else:
            self.z_nodes = np.asarray(z_nodes, dtype=float).copy()
        self.y = PchipInterpolator(self.x_nodes, self.path_center)
        self.z = PchipInterpolator(self.x_nodes, self.z_nodes)
        self.dy, self.ddy = self.y.derivative(), self.y.derivative(2)
        self.grid_x = np.linspace(self.x_nodes[0], self.x_nodes[-1],
                                  max(2, int(np.ptp(self.x_nodes)*20)+1))
        ds = np.sqrt(1+self.dy(self.grid_x)**2)
        self.grid_s = np.r_[self.x_nodes[0], self.x_nodes[0] +
                            np.cumsum(np.diff(self.grid_x)*(ds[1:]+ds[:-1])/2)]
        self.path_distance = np.interp(self.x_nodes, self.grid_x, self.grid_s)
        self.rail_coeff = np.polyfit(self.x_nodes, self.z_nodes, 1)
        self.observed_start, self.observed_end = self.grid_s[[0,-1]]

    def center_y(self, x):
        return self.y(np.clip(x, self.x_nodes[0], self.x_nodes[-1]))

    def _lut(self, step=.02):
        """Dense tables of the spline (and its derivatives) for fast per-point lookup.

        Linear interpolation on a 2 cm grid deviates from the cubic by well under
        a micrometre for any realistic tunnel curvature, but is ~10x cheaper than
        evaluating PCHIP for the ~250k points of a frame."""
        lut = getattr(self, '_lut_cache', None)
        if lut is None:
            x = np.arange(self.x_nodes[0], self.x_nodes[-1] + step, step)
            x = np.append(x, x[-1] + step)
            xc = np.clip(x, self.x_nodes[0], self.x_nodes[-1])
            tabs = np.column_stack((self.y(xc), self.dy(xc), self.ddy(xc), self.z(xc),
                                    np.interp(xc, self.grid_x, self.grid_s)))
            dtabs = np.diff(tabs, axis=0, append=tabs[-1:])
            # column-contiguous copies: gathering from strided columns is ~3x slower
            lut = (float(self.x_nodes[0]), 1.0 / step,
                   [np.ascontiguousarray(tabs[:, c]) for c in range(tabs.shape[1])],
                   [np.ascontiguousarray(dtabs[:, c]) for c in range(dtabs.shape[1])])
            self._lut_cache = lut
        return lut

    def to_track(self, points):
        if not len(points):
            return np.empty((0,3))
        x0, inv, tabs, dtabs = self._lut()
        n = len(tabs[0]) - 2

        def look(x, cols):
            t = (x - x0) * inv
            i = np.minimum(t.astype(np.intp), n)
            f = t - i
            return [tabs[c][i] + dtabs[c][i] * f for c in cols]

        px, py, pz = points[:,0], points[:,1], points[:,2]
        x = np.clip(px, self.x_nodes[0], self.x_nodes[-1])
        # One Newton step stays within five micrometres of the converged
        # projection even at a 150 m curve radius.
        yv, slope, ddy = look(x, (0, 1, 2))
        residual = yv - py
        gradient = x - px + residual*slope
        hessian = 1 + slope*slope + residual*ddy
        x = np.clip(x - np.clip(gradient/np.maximum(hessian, .2), -3, 3),
                    self.x_nodes[0], self.x_nodes[-1])
        yv, slope, zv, sv = look(x, (0, 1, 3, 4))
        norm = np.sqrt(1 + slope*slope)
        dx = px - x; dy = py - yv
        lateral = (-slope*dx + dy)/norm
        along = (dx + slope*dy)/norm
        return np.column_stack((sv + along, lateral, pz - zv))

    def to_base(self, points):
        x=np.interp(points[:,0],self.grid_s,self.grid_x)
        slope=self.dy(x); norm=np.sqrt(1+slope*slope)
        return np.column_stack((x-points[:,1]*slope/norm,
                                self.y(x)+points[:,1]/norm,self.z(x)+points[:,2]))


def fit_rails(points, gauge=1.52, max_distance=80., previous=None, presorted=False):
    points=np.asarray(points)
    if len(points)<100:
        return None, []
    # Sort once. Window follows the predicted rail centre, not the LiDAR axis.
    if not presorted:
        points=points[np.argsort(points[:,0])]
    observations=[]; selected_left=[]; selected_right=[]
    prior_y=0.; prior_z=None; slope=0.; misses=0
    previous_x=4.5
    for middle in np.arange(4.5, max_distance-1., 5.):
        predicted=prior_y+slope*(middle-previous_x)
        if previous is not None and previous.x_nodes[0]<=middle<=previous.x_nodes[-1]:
            old_center=float(previous.center_y(middle))
            if abs(old_center-predicted)<.75:
                predicted=.65*old_center+.35*predicted
        half_length = 2.5
        i,j=np.searchsorted(points[:,0],[middle-half_length-1.,middle+half_length+1.])
        local=points[i:j]
        if len(local)<30:
            if observations:
                misses+=1
                if misses>=3:
                    break
            continue
        norm=np.sqrt(1+slope*slope)
        lateral=(local[:,1]-predicted-slope*(local[:,0]-middle))/norm
        longitudinal=((local[:,0]-middle)+slope*(local[:,1]-predicted))/norm
        keep=(np.abs(lateral)<1.8)&(np.abs(longitudinal)<half_length)
        local=local[keep]; lateral=lateral[keep]
        if len(local)<12 or np.max(local[:,0], initial=-np.inf)<middle:
            if observations:
                misses+=1
                if misses>=3:
                    break
            continue
        z_edges=np.arange(-4.5,2.01,.04)
        hist,_=np.histogram(local[:,2],z_edges)
        zcent=(z_edges[1:]+z_edges[:-1])/2
        peaks=np.argsort(hist)[-14:].tolist()
        if prior_z is not None:
            peaks=[k for k in peaks if abs(zcent[k]-prior_z)<.36]
        if peaks:
            zscale=max(hist[k] for k in peaks)
            chosen=max(peaks,key=lambda k:hist[k]/max(zscale,1)-
                       (1.4*abs(zcent[k]-prior_z) if prior_z is not None else 0))
            peaks=[chosen]
        best=None
        for k in peaks:
            rail=(np.abs(local[:,2]-zcent[k])<.14)
            yy=lateral[rail]
            edges=np.arange(-1.8,1.801,.05)
            counts,_=np.histogram(yy,edges)
            centers=(edges[:-1]+edges[1:])/2
            pp=np.argsort(counts)[-20:]
            L=np.repeat(pp,len(pp)); R=np.tile(pp,len(pp))     # same (l outer, r inner) order as the loops
            gap=centers[R]-centers[L]
            minimum_support = 1 if middle>40 else 2
            offset=(centers[L]+centers[R])/2
            ok=((gap>1.20)&(gap<1.75)&(np.minimum(counts[L],counts[R])>=minimum_support)&
                (np.abs(offset)<=(.6 if observations else .8)))
            if ok.any():
                L,R,gap,offset=L[ok],R[ok],gap[ok],offset[ok]
                support=(counts[L]+counts[R])/max(1,counts.max())
                score=support-1.7*np.abs(gap-(gauge-.20))-4*np.abs(offset)
                m=int(np.argmax(score))
                if best is None or score[m]>best[0]:
                    best=(float(score[m]),predicted+offset[m]*norm,zcent[k],float(gap[m]),
                          float(counts[L[m]]+counts[R[m]]), centers[L[m]], centers[R[m]])
        if best is None:
            misses+=1
            if misses>=3: break
            continue
        misses=0
        _, center, zvalue, measured_gauge, support, left_peak, right_peak=best
        if observations:
            measured_slope=(center-prior_y)/(middle-previous_x)
            # Bound changes in heading, not absolute lateral displacement.
            if abs(measured_slope-slope)>.18: break
        observations.append((middle,center,zvalue,measured_gauge,support))
        # Keep actual input returns contributing to each selected histogram bin.
        # Reconstructed rail lines alone cannot establish correct association.
        at_height=np.abs(local[:,2]-zvalue)<.14
        selected_left.append(local[at_height&(lateral>=left_peak-.025)&(lateral<left_peak+.025)].copy())
        selected_right.append(local[at_height&(lateral>=right_peak-.025)&(lateral<right_peak+.025)].copy())
        if len(observations)>=4:
            recent=np.asarray(observations[-6:])
            t=recent[:,0]-middle
            coef=np.polyfit(t,recent[:,1],2)
            residual=np.sqrt(np.mean((np.polyval(coef,t)-recent[:,1])**2))
            slope=float(np.polyval(np.polyder(coef),0.)) if residual<.055 else 0.
        prior_y=predicted+.3*(center-predicted)
        prior_z,previous_x=zvalue,middle
    if len(observations)<4:
        return None, observations
    model=PathModel(observations)
    if previous is not None:
        # Register the old path to fresh rail observations before using it for
        # stabilization.  For a smooth curve, forward ego motion appears as an
        # approximately affine lateral correction (offset + heading change).
        # Directly averaging sensor-frame coordinates would make the corridor
        # lag behind the turn.
        fresh=np.asarray(observations,float)
        overlap=(fresh[:,0]>=previous.x_nodes[0])&(fresh[:,0]<=previous.x_nodes[-1])
        if np.count_nonzero(overlap)>=4:
            x=fresh[overlap,0]
            old_y=previous.center_y(x)
            weights=np.sqrt(np.clip(fresh[overlap,4],2.,80.))
            # Register against measurements, not a spline whose end condition
            # changes whenever the visible range changes. Huber reweighting
            # prevents one sparse, misidentified rail pair rotating the path.
            target=fresh[overlap,1]-old_y
            robust=np.ones(len(x))
            for _ in range(4):
                correction=np.polyfit(x,target,1,w=weights*np.sqrt(robust))
                errors=target-np.polyval(correction,x)
                robust=np.minimum(1.,.10/np.maximum(np.abs(errors),1e-6))
            aligned=old_y+np.polyval(correction,x)
            residual=target-np.polyval(correction,x)
            alignment_rms=np.sqrt(np.average(residual*residual,weights=weights**2*robust))
            if alignment_rms<.18:
                current_weight=np.clip(fresh[overlap,4]/(fresh[overlap,4]+20.),.10,.8)
                fresh[overlap,1]=(current_weight*model.center_y(x)+
                                  (1-current_weight)*aligned)
                model=PathModel(fresh)
    model.selected_left=np.concatenate(selected_left) if selected_left else np.empty((0,3))
    model.selected_right=np.concatenate(selected_right) if selected_right else np.empty((0,3))
    return model,observations
