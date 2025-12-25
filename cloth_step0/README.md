# Cloth Step 0 — GPU Integration (WebGPU Compute)

Goal: validate the compute pipeline by updating positions with constant velocities:

pos[i] = pos[i] + vel[i] * dt

Concepts used:
- storage buffers (positions, velocities)
- uniform buffer (dt, n)
- bind group layout + bind group
- compute pipeline
- dispatch_workgroups + queue.submit
