# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:28:54 2022

@author: mehran
"""
import math
r = 1000/(2*math.pi)

x, y = 0, 0
c = 1500
width = 2.5
with open("circle_3.edg.xml", "w") as output:
    angle = 2 * math.pi / c
    shape = ["%.2f,%.2f" % (math.cos(i * angle) * r + x,
                            math.sin(i * angle) * r + y) for i in range(c)]
    print('''
<edges>
    <edge id="e1" from="n1" to="n2" width="10.2" speed="60" shape="%s"/>
    <edge id="e2" from="n2" to="n1" width="10.2" speed="60"/>
</edges>''' % " ".join(shape), file=output)



128.76,-93.55
with open("circle.nod.xml", "w") as output:
    print('''
<nodes>
    <node id="n1" x="159.15" y="0.00" />
    <node id="n2" x="128.76" y="-93.55" />
</nodes>''', file=output)


with open("circle_2.nod.xml", "w") as output:
    print('''
<nodes>
    <node id="n1" x="63.66" y="0.00" />
    <node id="n2" x="51.19" y="-37.85" />
</nodes>''', file=output)

# Shape e2:
#shape=" 151.57,-48.55 151.77,-47.91 151.97,-47.28 152.17,-46.64 152.36,-46.00 152.55,-45.36 152.74,-44.72 152.93,-44.08 153.11,-43.44 153.29,-42.80 153.47,-42.16 153.65,-41.51 153.82,-40.87 153.99,-40.23 154.15,-39.58 154.32,-38.93 154.48,-38.29 154.64,-37.64 154.80,-36.99 154.95,-36.34 155.10,-35.69 155.25,-35.04 155.39,-34.39 155.54,-33.74 155.68,-33.09 155.81,-32.44 155.95,-31.78 156.08,-31.13 156.21,-30.48 156.34,-29.82 156.46,-29.17 156.58,-28.51 156.70,-27.86 156.81,-27.20 156.93,-26.54 157.04,-25.88 157.14,-25.23 157.25,-24.57 157.35,-23.91 157.45,-23.25 157.54,-22.59 157.64,-21.93 157.73,-21.27 157.82,-20.61 157.90,-19.95 157.98,-19.29 158.06,-18.62 158.14,-17.96 158.21,-17.30 158.28,-16.64 158.35,-15.97 158.42,-15.31 158.48,-14.65 158.54,-13.98 158.60,-13.32 158.65,-12.65 158.70,-11.99 158.75,-11.32 158.80,-10.66 158.84,-9.99 158.88,-9.33 158.92,-8.66 158.95,-8.00 158.99,-7.33 159.02,-6.66 159.04,-6.00 159.07,-5.33 159.09,-4.67 159.10,-4.00 159.12,-3.33 159.13,-2.67 159.14,-2.00 159.15,-1.33 159.15,-0.67 159.15,0.00"


#netconvert --node-files=circle_2.nod.xml --edge-files=circle_2.edg.xml --output-file=circle_2.net.xml



with open("circle_2.rou.xml", "w") as output:
    print('''
<routes>
    <vType id="type1" accel="5" decel="7.5" laneChangeModel="SL2015" lcPushyGap=".2" sigma="0.5" length="3.2" maxSpeed="30" latAlignment="compact" minGapLat=".1" tau=".01"  departPosLat="center"/>
    <vType id="type2" accel="3.5" decel="3.5" laneChangeModel="SL2015" lcPushyGap=".2" sigma="0.01" length="3.2" width="1.8" latAlignment='arbitrary' minGapLat=".5" tau=".5" maxSpeed="40" minGap=".5"  departPosLat="center" maxSpeedLat="5" lcAccelLat="5000"/>
    <vType id="type3" accel="0.8" decel="3.5" sigma="0.5" length="5"  maxSpeed="30" latAlignment="compact" minGapLat=".1" tau=".1" maxSpeedLat="1" lcAccelLat="5000"/>
    <route id="route0"  edges="e1 e2"/>
</routes> ''', file=output)

with open("circle_2.add.xml", "w") as output:
    print('''
<additionals>
    <rerouter id="rerouter_0" edges="e1">
        <interval end="1e9">
           <destProbReroute id="e2"/>
        </interval>
    </rerouter>
    <rerouter id="rerouter_1" edges="e2">
        <interval end="1e9">
           <destProbReroute id="e1"/>
        </interval>
    </rerouter>
</additionals> ''', file=output)


with open("circle_2.sumocfg", "w") as output:
    print('''
<configuration>
    <input>
       <net-file value="circle_2.net.xml"/>
       <route-files value="circle_2.rou.xml"/>
       <lateral-resolution value=".2"/>
       <collision.action value="warn"/>
       <additional-files value="circle_2.add.xml"/>
    </input>
    <time>
	<step-length value= ".25" />
    </time>
</configuration> ''', file=output)

















