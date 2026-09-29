"""Follow the raised rail profiles, rather than peaks on the trackbed plane."""
import numpy as np
from scipy.signal import find_peaks


def fit_heads(points, seed, model_class, gauge=1.52, maximum=80., presorted=False):
    points=np.asarray(points)
    if not presorted:
        points=points[np.argsort(points[:,0])]
    xx=np.ascontiguousarray(points[:,0])
    observations=[];left_returns=[];right_returns=[]
    center=0.;slope=0.;curvature=0.;last_x=4.5;misses=0
    edges=np.arange(-2.,2.001,.04)
    centers=(edges[1:]+edges[:-1])/2
    for middle in np.arange(4.5,maximum-1.,5.):
        dx=middle-last_x
        predicted=center+slope*dx+curvature*dx*dx
        heading=slope+2*curvature*dx
        i,j=np.searchsorted(xx,[middle-2.5,middle+2.5])
        local=points[i:j]
        if len(local)<12:
            misses+=1
            if misses>=3:
                break
            continue
        norm=np.sqrt(1+heading*heading)
        lateral=(local[:,1]-predicted-heading*(local[:,0]-middle))/norm
        height=local[:,2]-seed.z(local[:,0])
        mask=(height>.08)&(height<.34)&(np.abs(lateral)<2.)
        head_points=local[mask];yy=lateral[mask]
        counts,_=np.histogram(yy,edges)
        smooth=np.convolve(counts,[.2,.6,.2],mode='same')
        prominence=.10 if middle>40. else .15
        peaks,_=find_peaks(smooth,distance=3,prominence=prominence)
        peaks=sorted(peaks,key=lambda k:-smooth[k])[:16]
        best=None
        if len(peaks)>=1:
            pk=np.asarray(peaks)
            L=np.repeat(pk,len(pk)); R=np.tile(pk,len(pk))     # same (l outer, r inner) order as the loops
            gap=centers[R]-centers[L]; offset=(centers[R]+centers[L])/2
            ok=(gap>gauge-.14)&(gap<gauge+.20)&(np.abs(offset)<.5)
            if ok.any():
                L,R,gap,offset=L[ok],R[ok],gap[ok],offset[ok]
                cs=np.concatenate(([0],np.cumsum(counts)))
                lc=cs[np.minimum(L+2,len(counts))]-cs[np.maximum(0,L-1)]
                rc=cs[np.minimum(R+2,len(counts))]-cs[np.maximum(0,R-1)]
                good=np.minimum(lc,rc)>=(3 if middle<25 else 1)
                if good.any():
                    L,R,gap,offset,lc,rc=L[good],R[good],gap[good],offset[good],lc[good],rc[good]
                    score=np.log1p(np.minimum(lc,rc))-3*np.abs(offset)-2*np.abs(gap-(gauge+.04))
                    k=int(np.argmax(score))
                    best=(score[k],float(offset[k]),float(gap[k]),int(L[k]),int(R[k]),int(lc[k]+rc[k]))
        if best is None:
            misses+=1
            if misses>=3:break
            continue
        misses=0
        _,offset,gap,l,r,support=best
        center=predicted+offset*norm
        observations.append((middle,center,float(seed.z(middle)),gap,support))
        left_returns.append(head_points[np.abs(yy-centers[l])<.06])
        right_returns.append(head_points[np.abs(yy-centers[r])<.06])
        recent=np.asarray(observations[-6:])
        t=recent[:,0]-middle
        degree=min(2,len(recent)-1)
        if degree:
            weights=np.sqrt(np.clip(recent[:,4],2.,30.))
            coef=np.polyfit(t,recent[:,1],degree,w=weights)
            for _ in range(3):
                residual=np.abs(recent[:,1]-np.polyval(coef,t))
                robust=np.minimum(1.,.08/np.maximum(residual,1e-6))
                coef=np.polyfit(t,recent[:,1],degree,w=weights*np.sqrt(robust))
            slope=float(np.polyval(np.polyder(coef),0.))
            curvature=float(np.clip(coef[0],-.004,.004)) if degree==2 else 0.
        last_x=middle
    if len(observations)<4:
        return None,observations
    model=model_class(observations)
    model.selected_left=np.concatenate(left_returns)
    model.selected_right=np.concatenate(right_returns)
    return model,observations
