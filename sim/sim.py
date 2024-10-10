import rebound

sim = rebound.Simulation()
sim.start_server(port=1234)
sim.add(m=1.)
sim.add(m=1e-3, a=1., e=0.1)
sim.add(a=1.4, e=0.1)
for p in sim.particles:
    print(p.x, p.y, p.z)
print("")

for i in range(10):
    sim.integrate(1000. * i)
    for p in sim.particles:
        print(p.x, p.y, p.z)
    print("")