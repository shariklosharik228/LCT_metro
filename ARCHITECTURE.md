# Architecture

```
rosbag ──▶ bag_player_node ──▶ /lidar_points ──▶ detector_node ──▶ /developer_obstacle/* ──▶ RViz
                                                    │
                                    metro_obstacle_core (без ROS)
  preprocess ─▶ rails (near) ─▶ walls (far) ─▶ track coords ─▶ ROI/clearance ─▶ accumulation
                                                               ─▶ clustering ─▶ shape checks ─▶ tracker
```

| Модуль | Назначение |
|---|---|
| `config.py` | все параметры алгоритма (каждый = ROS-параметр с тем же именем) |
| `rail_geometry.py`, `rail_heads.py` | плоскость пути и поднятые головки рельсов → ось пути вблизи (≈50–70 м) |
| `wall_corridor.py` | продолжение оси по стенам тоннеля до 200–300 м + короткая проверяемая экстраполяция |
| `ego_motion.py` | скорость поезда по смещению выступов стены (кронштейны, короба), без одометрии |
| `tracking.py` | многообъектный трекер: подтверждение k из n кадров, проверка устойчивости положения |
| `detector.py` | конвейер кадра, классификация кластеров, итоговый результат |
| `simulation.py` | ray-cast симулятор тоннеля (Hesai-128-подобная развёртка) для тестов и экспериментов |

`metro_obstacle_ros`: `detector_node` (PointCloud2 → алгоритм → топики/маркеры),
`bag_player_node` (обёртка над `ros2 bag play`).
