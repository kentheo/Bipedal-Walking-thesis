import sys, math
import numpy as np
import matplotlib.pyplot as plt

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.

#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 Time steps.
#
# To solve hardcore version you need 300 points in 2000 Time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 100
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 1000
SPEED_HIP     = 12
SPEED_KNEE    = 15
LIDAR_RANGE   = 160/SCALE

HULL_POLY =[
    (-10,+10), (+10,+10),
    (+10,-10), (-10,-10),
    ]

LEG_W, LEG_H = 6/SCALE, 34/SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

# adjustable constants
g = 10        # gravity acceleration
T_step = 0.4    # step Time (s)
x_i = 8/SCALE       # initial position of CoM wrt the standing point
h_c = 2*LEG_H*0.9       # constant CoM height
v_i = 0/SCALE           # initial velocity

T_c = math.sqrt(h_c/g)    # Time constant

def geometry_sp(h, x, leg):
    # get angles of supporting leg
    L1 = math.sqrt(h**2+x**2)
    a1 = math.atan(x/h)
    if L1/2/leg > 1:
        a2 = math.acos(1)
    else:
        a2 = math.acos((L1/2)/leg)
    gamma = a2 - a1
    theta = math.pi - 2*a2
    return theta, gamma

theta_i, gamma_i = geometry_sp(h_c, x_i, LEG_H)
a_i = theta_i + gamma_i - math.pi/2

init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
init_y = TERRAIN_HEIGHT

hull_x = init_x + x_i
hull_y = init_y + h_c

leg_x = hull_x + LEG_H * math.sin(gamma_i) / 2
leg_y = hull_y - LEG_H * math.cos(gamma_i) / 2

lower_x = init_x + LEG_H * math.cos(a_i)/2
lower_y = init_y + LEG_H * math.sin(a_i)/2


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True
    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False

class BipedalWalker(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self):
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World(gravity=(0,0), doSleep=True)
        self.terrain = None
        self.hull = None

        self.prev_shaping = None
        self._reset()

        high = np.array([np.inf]*24)
        self.action_space = spaces.Box(np.array([-1,-1,-1,-1]), np.array([+1,+1,+1,+1]))
        self.observation_space = spaces.Box(-high, high)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity=0 #+= self.np_random.uniform(-1, 1)/SCALE   #1
                y += velocity
                #y = TERRAIN_HEIGHT

            elif state==PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x,              y),
                    (x+TERRAIN_STEP, y),
                    (x+TERRAIN_STEP, y-4*TERRAIN_STEP),
                    (x,              y-4*TERRAIN_STEP),
                    ]
                t = self.world.CreateStaticBody(
                    fixtures = fixtureDef(
                        shape=polygonShape(vertices=poly),
                        friction = FRICTION
                    ))
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                t = self.world.CreateStaticBody(
                    fixtures = fixtureDef(
                        shape=polygonShape(vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]),
                        friction = FRICTION
                    ))
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*TERRAIN_STEP

            elif state==STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                    ]
                t = self.world.CreateStaticBody(
                    fixtures = fixtureDef(
                        shape=polygonShape(vertices=poly),
                        friction = FRICTION
                    ))
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state==STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        ]
                    t = self.world.CreateStaticBody(
                        fixtures = fixtureDef(
                            shape=polygonShape(vertices=poly),
                            friction = FRICTION
                        ))
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height)*TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            t = self.world.CreateStaticBody(
                fixtures = fixtureDef(
                    shape=edgeShape(vertices=poly),
                    friction = FRICTION,
                    categoryBits=0x0001,
                ))
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def _reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # self._generate_terrain(self.hardcore)
        self._generate_clouds()

        self.hull = self.world.CreateDynamicBody(
            position = (hull_x, hull_y),
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (0.3,0.3,0.5)
        #self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        #self.hull.ApplyForceToCenter((0, 10*10.0*5.0*(20/SCALE)**2), False)

        # Fixed "terrain point" to pin hull
        poly_x = TERRAIN_STEP*3
        poly_y = TERRAIN_HEIGHT+ 2/SCALE
        
        poly = [
                    (hull_x,              hull_y),
                    (hull_x+50/SCALE, hull_y),
                    (hull_x+50/SCALE, hull_y+50/SCALE),
                    (hull_x,              hull_y+50/SCALE),
                    ]
        t = self.world.CreateStaticBody(
            fixtures = fixtureDef(
                shape=polygonShape(vertices=poly),
                friction = FRICTION
            ))
        t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
        print(poly)
        # Joint to join the terrain point with the hull
        pin = revoluteJointDef(
                bodyA=t,
                bodyB=self.hull,
                localAnchorA=(0, hull_y-25/SCALE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=0.0,
                motorSpeed = 0.0,
                referenceAngle = 0.0,
                lowerAngle = 1.4,      # -80 degree
                upperAngle = 1.4,       # 80 degree
                )
        self.world.CreateJoint(pin)

        self.legs = []
        self.joints = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (leg_x, leg_y),
                angle = gamma_i,
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )

            leg.color1 = (0.9-i/10., 0.3-i/10., 0.5-i/10.)
            leg.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 0.0,
                referenceAngle = 0.0,
                lowerAngle = -3.14,      # -80 degree
                upperAngle = 3.14,       # 80 degree
                )

            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position = (lower_x, lower_y),
                angle = (-math.pi/2+a_i),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H/2),
                localAnchorB=(0, LEG_H/2),  
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 0.0,
                referenceAngle = -1*math.pi,
                lowerAngle = 3.1399, #0.35,    # 20 degree
                upperAngle = 3.1401, #3.14,    # 180 degree
                )

            lower.ground_contact = False
            # foot.ground_contact = False

            self.legs.append(lower)
            # self.legs.append(foot)
            self.joints.append(self.world.CreateJoint(rjd))
            # self.joints.append(self.world.CreateJoint(foot_jd))

        self.drawlist = self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(10)]

        #self.hull.ApplyForceToCenter((0, 10.0 * (5.0 * (20 / SCALE) ** 2)), True)
        #self.legs[0].ApplyForceToCenter((0, 10.0 * 1.0 * LEG_H * LEG_W), True)
        #self.legs[1].ApplyForceToCenter((0, 10.0 * 1.0 * LEG_H * LEG_W), True)
        #self.legs[2].ApplyForceToCenter((0, 10.0 * 1.0 * LEG_H * LEG_W), True)
        #self.legs[3].ApplyForceToCenter((0, 10.0 * 1.0 * LEG_H * LEG_W), True)

        return self._step(np.array([0.0,0.0,0.0,0.0]))[0]

    def _step(self, action):

        #if action[0]==action[1]==action[2]==action[3]==0.0:
        #    self.hull.ApplyForceToCenter((0, 10.0*(5.0*(20/SCALE)**2)), True)
        #    self.legs[0].ApplyForceToCenter((0, 10.0 * 1.0 * LEG_H*LEG_W), True)
        #    self.legs[1].ApplyForceToCenter((0, 10.0 * 1.0 * LEG_H * LEG_W), True)
        #    self.legs[2].ApplyForceToCenter((0, 10.0 * 1.0 * LEG_H * LEG_W), True)
        #    self.legs[3].ApplyForceToCenter((0, 10.0 * 1.0 * LEG_H * LEG_W), True)

        #self.hull.ApplyForceToCenter((0, 20), True) #-- Uncomment this to receive a bit of stability help
        control_speed = False
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))

        else:
            self.joints[0].maxMotorTorque = float(np.abs(action[0]))
            self.joints[0].motorSpeed     = float(SPEED_HIP     * np.sign(action[0]))

            self.joints[1].maxMotorTorque = float(np.abs(action[1]))
            self.joints[1].motorSpeed     = float(SPEED_KNEE    * np.sign(action[1]))

            self.joints[2].maxMotorTorque = float(np.abs(action[2]))
            self.joints[2].motorSpeed     = float(SPEED_HIP     * np.sign(action[2]))

            self.joints[3].maxMotorTorque = float(np.abs(action[3]))
            self.joints[3].motorSpeed     = float(SPEED_KNEE    * np.sign(action[3]))

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,            #s[0]
            self.hull.angularVelocity,  #s[1]
            vel.x,                      #s[2]   global dxcom
            vel.y,                      #s[3]
            self.joints[0].angle,       #s[4]
            self.joints[0].speed,       #s[5]
            self.joints[1].angle,       #s[6]
            self.joints[1].speed,       #s[7]
            1.0 if self.legs[1].ground_contact else 0.0,    #s[8]   
            self.joints[2].angle,                           #s[9]
            self.joints[2].speed,                           #s[10]
            self.joints[3].angle,                           #s[11]
            self.joints[3].speed,                           #s[12]
            1.0 if self.legs[3].ground_contact else 0.0,     #s[13]  
            pos.y - TERRAIN_HEIGHT,                          #s[14]
            pos.x - self.legs[1].position.x,    # s[15]
            pos.x - self.legs[3].position.x,     # s[16]
            pos.x - init_x,   # global xcom      s[17]
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state)==28

        self.scroll = pos.x - VIEWPORT_W/SCALE/5

        shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True
        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        # for poly, color in self.terrain_poly:
        #     if poly[1][0] < self.scroll: continue
        #     if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
        #     self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render+1) % 100
        i = self.lidar_render
        if i < 2*len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
            self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        # flagy1 = TERRAIN_HEIGHT
        # flagy2 = flagy1 + 50/SCALE
        # x = TERRAIN_STEP*3
        # self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        # f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        # self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        # self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class BipedalWalkerHardcore(BipedalWalker):
    hardcore = True





# Main Function
############################
# Characters Explanation ----   s: start of each step    e:end of each step     x:distance wrt supporting point
#                               v:velocity of CoM         i:initial condition when all bodies are initialized
#                               est: estimated (according to LIPM)    mv: moving leg   sp:supporting leg
# For online estimation -----
#            key variables:     P: foot placement wrt CoM;


if __name__=="__main__":

    env = BipedalWalker()
    env.reset()
    steps = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    RISING, FALLING = 1, 2
    stage = RISING
    moving_leg = 0
    supporting_leg = 1 - moving_leg

    T_m = T_step / 2        # the moving leg will reach the highest point of its trajectory at half the step Time

    # initial condition of the first step
    x_s = x_i
    v_s = v_i

    v_desired = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]    # target CoM velocity after each step (m/s)
    j = 0

    x_ankle_s = 0       # start point of the moving leg polynomial trajectory
    d = x_i         # distance between supporting point and CoM after leg exchange

    # lists for plotting
    gamma_targ_0, gamma_real_0 = [], []
    knee_targ_0, knee_real_0 = [], []
    gamma_targ_1, gamma_real_1 = [], []
    knee_targ_1, knee_real_1 = [], []
    torque_a0, torque_a1, torque_a2, torque_a3 = [],[],[],[]
    hull_real, hull_targ = [], []
    vel_real, vel_targ = [], []
    height_real, height_targ = [], []
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    s7 = []
    s10 = []
    s12 = []

    distance = []

    vel_list = []
    P_list = []

    step_num = 0

    hip1_vel = []
    hip2_vel = []
    knee1_vel = []
    knee2_vel = []
    s6 = []

    # For posture control,  To be accumulated during stance
    delta_hip_angle = 0.0

    # Initialisation for filter
    x_ft = np.zeros((4,2))
    raw_data = np.zeros((4,2))

    hip1_vel_filter = []
    hip2_vel_filter = []
    knee1_vel_filter = []
    knee2_vel_filter = []
    hip_todo_plot = []
    hip_targ_plot = []

    hip1_ang_filter = []
    hip2_ang_filter = []
    knee1_ang_filter = []
    knee2_ang_filter = []

    while True:
        t = (steps)/float(FPS)

        # READ: estimate CoM position and velocity
        x_t_est = x_s*math.cosh(t/T_c) + T_c*v_s*math.sinh(t/T_c)     # estimated CoM velocity by LIPM
        v_t_est = x_s/T_c*math.sinh(t/T_c) + v_s*math.cosh(t/T_c)
        x_T_est = x_s * math.cosh(T_step / T_c) + T_c * v_s * math.sinh(T_step / T_c)
        v_T_est = x_s / T_c * math.sinh(T_step / T_c) + v_s * math.cosh(T_step / T_c)

        #vel_targ.append(v_t_est)
        vel_targ.append(v_t_est/2)

        # READ: IK to obtain hip (gamma) and knee (theta) angles of supporting leg
        theta_sp_est, gamma_sp_est = geometry_sp(h_c, x_t_est, LEG_H)

        # Foot placement estimation
        P = T_c * v_T_est * (math.cosh(T_step / T_c) / math.sinh(T_step / T_c)) \
            - T_c * v_desired[j] / math.sinh(T_step / T_c)


        s, r, done, info = env.step(a)
        steps += 1

        distance.append(s[15+supporting_leg])

        #print s[14]

        # Append to lists for plotting
        vel_real.append(s[2]/2)
        s3.append(s[3])
        s5.append(s[5])
        s7.append(s[7])
        s10.append(s[10])
        s12.append(s[12])

        contact_flag = [s[8], s[13]]

        hip_targ  = [0.0, 0.0]
        knee_targ = [0.0, 0.0]
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        hip_targ[supporting_leg] = gamma_sp_est - s[0]
        knee_targ[supporting_leg] = theta_sp_est

        # moving leg polynomial trajectory parameters
        x_ankle_e = x_T_est + P
        z_ankle_s = 0
        z_ankle_e = 0
        z_ankle_m = h_c / 5  # max height

        if stage==RISING:

            # polynomial trajectory
            x_ankle_t = x_ankle_s + 3 * (x_ankle_e - x_ankle_s) * (t ** 2) / (T_step ** 2) \
                        - 2 * (x_ankle_e - x_ankle_s) * (t ** 3) / (T_step ** 3)
            z_ankle_t = z_ankle_s + 3 * (z_ankle_m - z_ankle_s) * (t ** 2) / (T_m ** 2) \
                        - 2 * (z_ankle_m - z_ankle_s) * (t ** 3) / (T_m ** 3)

            # READ: IK to obtain hip and joint
            virtual_leg = math.sqrt((h_c-z_ankle_t)**2+((d+x_t_est-x_s)-(x_ankle_t-x_ankle_s))**2)
            if (virtual_leg/2)/LEG_H >1:
                theta_mv_est = 2 * math.asin(1)
            else:
                theta_mv_est = 2*math.asin((virtual_leg/2)/LEG_H)
            b1 = math.asin((h_c-z_ankle_t)/virtual_leg)
            b2 = (math.pi-theta_mv_est)/2

            if (x_ankle_t-x_ankle_s) <= (d+x_t_est-x_s):
                gamma_mv_est = b1 + b2 - math.pi/2
            else:
                gamma_mv_est = math.pi/2 - b1 + b2

            hip_targ[moving_leg] = gamma_mv_est - s[0]
            knee_targ[moving_leg] = theta_mv_est

        else:
            x_ankle_t = x_ankle_s + 3 * (x_ankle_e - x_ankle_s) * (t ** 2) / (T_step ** 2) \
                        - 2 * (x_ankle_e - x_ankle_s) * (t ** 3) / (T_step ** 3)
            z_ankle_t = z_ankle_m + 3 * (z_ankle_e - z_ankle_m) * ((t-T_m) ** 2) / ((T_step-T_m) ** 2) \
                        - 2 * (z_ankle_e - z_ankle_m) * ((t-T_m) ** 3) / ((T_step-T_m) ** 3)
            # moving leg geometry
            virtual_leg = math.sqrt((h_c-z_ankle_t)**2+((d+x_t_est-x_s)-(x_ankle_t-x_ankle_s))**2)
            if (virtual_leg/2)/LEG_H >1:
                theta_mv_est = 2 * math.asin(1)
            else:
                theta_mv_est = 2*math.asin((virtual_leg/2)/LEG_H)
            b1 = math.asin((h_c-z_ankle_t)/virtual_leg)
            b2 = (math.pi-theta_mv_est)/2
            if (x_ankle_t - x_ankle_s) <= (d + x_t_est-x_s):
                gamma_mv_est = b1 + b2 - math.pi / 2
            else:
                gamma_mv_est = math.pi / 2 - b1 + b2

            hip_targ[moving_leg] = gamma_mv_est - s[0]
            knee_targ[moving_leg] = theta_mv_est

        ############
        if t >= T_step / 2:  # after half of the step Time, lower leg will begin fall down
           stage = FALLING

        gamma_targ_0.append(hip_targ[0]+s[0])
        gamma_targ_1.append(hip_targ[1]+s[0])
        knee_targ_0.append(knee_targ[0])
        knee_targ_1.append(knee_targ[1])
        gamma_real_0.append(s[4]+s[0])
        gamma_real_1.append(s[9]+s[0])
        knee_real_0.append(s[6])
        knee_real_1.append(s[11])

        #print gamma_real_0, gamma_real_1, knee_real_0, knee_real_1

        # Posture Control PD
        desired_angle = math.sin(steps/FPS*2)
        hull_targ.append(desired_angle)

        # K_p = 65.0
        # K_d = 0.75
        # actual_angle_body = s[0]
        # angular_velocity_body = s[1]
        # # current_hip_targ = [hip_targ[0] - s[4], hip_targ[1] - s[9]]

        # angularVelocity_d = K_p * (desired_angle - actual_angle_body) - (K_d * angular_velocity_body)

        # delta_hip_angle += angularVelocity_d*1.0/FPS

        # print("Before0: " + str(hip_targ[0]), "Before1: " + str(hip_targ[1]))
        # hip_targ[0] -= angularVelocity_d*t
        # hip_targ[1] -= angularVelocity_d*t
        # print("After0: " + str(hip_targ[0]), "After1: " + str(hip_targ[1]))

        # print("Accumulated delta_hip_angle: " + str(delta_hip_angle) + " at supporting_leg: " + str(supporting_leg) + " at stage: " + str(stage) + " at steps: " + str(steps))

        #hip_targ[supporting_leg] += delta_hip_angle

        hip_targ[0] = desired_angle #-delta_hip_angle
        # hip_targ[1] = desired_angle #-delta_hip_angle
        # knee_targ[0] = desired_angle #-delta_hip_angle
        # knee_targ[1] = desired_angle #-delta_hip_angle


        # Create the raw data matrix: hip, knee, hip, knee
        raw_data[0][0] = s[4]  # Hip1 angle
        raw_data[0][1] = s[5]  # Hip1 vel
        raw_data[1][0] = s[6]  # Knee1 angle
        raw_data[1][1] = s[7]  # Knee1 vel
        raw_data[2][0] = s[9]  # Hip2 angle
        raw_data[2][1] = s[10] # Hip2 vel
        raw_data[3][0] = s[11] # Knee2 angle
        raw_data[3][1] = s[12] # Knee2 vel

        # Filter raw data
        x_ft += 7.5*(raw_data - x_ft) * 1.0/FPS

        hip1_vel_filter.append(x_ft[0][1])
        hip2_vel_filter.append(x_ft[2][1])
        knee1_vel_filter.append(x_ft[1][1])
        knee2_vel_filter.append(x_ft[3][1])

        hip1_ang_filter.append(x_ft[0][0])
        hip2_ang_filter.append(x_ft[2][0])
        knee1_ang_filter.append(x_ft[1][0])
        knee2_ang_filter.append(x_ft[3][0])

        # PD Controllers
        hip_todo[0] = 22.5*(hip_targ[0] - s[4]) - 5.0*x_ft[0][1] #s[5]  # hip angle/hip speed leg 1
        hip_todo[1] = 2.5*(hip_targ[1] - s[9]) - 5.0*x_ft[2][1] #s[10] # hip angle/hip speed leg 2
        # hip_todo[0] = 1800.0*(hip_targ[0] - s[4]) - 0.1*s[5]  # hip angle/hip speed leg 1
        # hip_todo[1] = 1800.0*(hip_targ[1] - s[9]) - 0.1*s[10] # hip angle/hip speed leg 2

        hip_todo_plot.append(hip_todo[0])
        hip_targ_plot.append(hip_targ[0])

        # print(supporting_leg)
        
        # In terms of supporting leg:
        # brown leg is 1
        # pink leg is 0

        # Previous "support hack" - manages torque
        # hip_todo[0] -= 1800.0*(0-s[0]) - 0.0*s[1]
        # hip_todo[1] -= 1800.0*(0-s[0]) - 0.0*s[1]

        knee_todo[0] = 5.0 * (knee_targ[0] - s[6]) - 1.5 * x_ft[1][1]#s[7]     # knee angle/knee speed leg 1
        knee_todo[1] = 5.0 * (knee_targ[1] - s[11]) - 1.5 * x_ft[3][1]#s[12]   # knee angle/knee speed leg 2
        # knee_todo[0] = 260.0 * (0 - s[6]) - 4.0 * s[7]     # knee angle/knee speed leg 1
        # knee_todo[1] = 260.0 * (0 - s[11]) - 4.0 * s[12]   # knee angle/knee speed leg 2


        # Previous knee hack
        # if s[0]>0.0:
        #     knee_todo[supporting_leg] += 200*(h_c - s[14])- 20.0*s[3]
        # if s[0]<0.0:
        #     knee_todo[supporting_leg] -= 200 * (h_c - s[14]) + 20.0 * s[3]

        # For plotting
        height_real.append(s[14]/2)

        s4.append(s[4])

        s6.append(s[6])


        hull_real.append(s[0])   # body angle
        torque_a0.append(a[0])
        torque_a1.append(a[1])
        torque_a2.append(a[2])
        torque_a3.append(a[3])

        hip1_vel.append(s[5]) # hip1 angular velocity
        hip2_vel.append(s[10]) # hip2 angular velocity
        knee1_vel.append(s[7]) # knee1 angular velocity
        knee2_vel.append(s[12]) # knee2 angular velocity


        a[0] = hip_todo[0]
        a[1] = 0.0
        a[2] = hip_todo[1]
        a[3] = 0.0
        env.render()

        # # READ: Leg exchange; Update CoM position and velocity (new initial condition)
        # if t >= T_step:
        # #if contact_flag[moving_leg] == 1.0:
        #     step_num += 1
        #     #h_c = s[14]

        #     x_s = -1 * P        # theoretical value

        #     v_s = v_T_est
        #     #v_s = s[2]/(0.3*(VIEWPORT_W / SCALE)/float(FPS))

        #     d = x_T_est
        #     x_ankle_s = -1 * x_ankle_e
        #     #a5_sp = a5_i - s[4+supporting_leg*5] - s[0]
        #     #theta_sp = theta_i + (s[6+supporting_leg*5]-1.0)
        #     #d = h_c / math.tan(math.pi - a5_sp - 0.5 * (math.pi - theta_sp))  # distance between CoM and suuport point
        #     #x_ankle_s = -1*(d-x_s)

        #     # leg exchange
        #     stage = RISING
        #     moving_leg = 1 - moving_leg
        #     supporting_leg = 1 - moving_leg
        #     #steps = 0

        #     # Clear to zero during swing
        #     #delta_hip_angle = 0.0
        #     print("At leg exchange: " + str(delta_hip_angle))

        #     #if j <= 4:
        #     #    j += 1

        #     #---------------------------------
        #     # Data Collection for Online Estimation
        #     #xcom.append(s[17])
        #     #dxcom.append(s[2])
        # #if step_num==12: break
        if steps >= FPS*10: break
        # if done: break

    z_targ = []

    # plot
    t_list = []
    # print("length of gamma targ 0: " + str(len(gamma_targ_0)))
    for j in range(len(gamma_targ_0)):
        t_list.append(float(j+1)/float(FPS))
        height_targ.append(h_c/2)
        #hull_targ.append(0)
        z_targ.append(0)
    plot_flag = "hip filters"

    if plot_flag=="hip":

        #plt.plot(t_list, gamma_targ_1, 'g',linewidth=2.0)
        #plt.plot(t_list, gamma_real_0, 'r',linewidth=2.0)

        plt.plot(t_list, knee_targ_1, 'g',linewidth=2.0)#, label="desired knee angle")
        #plt.plot(t_list, knee_real_0, 'r',linewidth=2.0, label="measured knee angle")
        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("Desired knee angle of leg 2 (rad)", fontsize=20)
        #plt.legend(loc="lower center", fontsize=20)

        #plt.figure(2)
        #plt.plot(t_list, gamma_targ_1, 'g', t_list, gamma_real_1, 'b')
        #plt.plot(t_list, gamma_targ_1, 'g')
        #plt.xlabel("Time (s)")
        #plt.ylabel("desired hip angle of Leg 2 (rad)")
        #plt.plot(t_list, hull_real, 'r', t_list, hull_targ, 'g')
        plt.grid()

    if plot_flag=="two":
        line1, = plt.plot(t_list, gamma_targ_0, 'g',linewidth=2.0, label="desired hip angle")
        line2, = plt.plot(t_list, gamma_real_0, 'r',linewidth=2.0, label="measured hip angle")

        line3, = plt.plot(t_list, knee_targ_0, 'y', linewidth=2.0,label="desired knee angle")
        line4, = plt.plot(t_list, knee_real_0, 'b', linewidth=2.0,label="measured knee angle")

        first_legend = plt.legend(handles=[line1, line2], loc="lower right")

        ax = plt.gca().add_artist(first_legend)
        plt.legend(handles=[line3, line4], loc="center right")

        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("Joint angles of Leg 1 (rad)", fontsize=20)

        plt.grid()

    if plot_flag=="knee":
        plt.figure(1)
        #plt.plot(t_list, knee_targ_0, 'g', t_list, knee_real_0, 'r')
        plt.plot(t_list, knee_targ_0, 'g')
        plt.xlabel("Time (s)")
        plt.ylabel("desired knee angle of Leg 1 (rad)")
        plt.grid()
        plt.figure(2)
        #plt.plot(t_list, knee_targ_1, 'g', t_list, knee_real_1, 'b')
        plt.plot(t_list, knee_targ_1, 'g')
        plt.xlabel("Time (s)")
        plt.ylabel("desired knee angle of Leg 2 (rad)")
        plt.grid()
    if plot_flag=="height":
        plt.plot(t_list, height_real, 'r', linewidth=2.0, label="measured CoM height")
        plt.plot(t_list, height_targ, 'g',linewidth=2.0, label="desired CoM height")
        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("Height of CoM (m)", fontsize=20)
        plt.legend(loc="lower center", fontsize=20)
        plt.grid()

    if plot_flag=="action":
        plt.plot(t_list, torque_a0, 'r')
        plt.plot(t_list, torque_a1, 'r')
        plt.plot(t_list, torque_a2, 'r')
        plt.plot(t_list, torque_a3, 'r')

    if plot_flag=="hull_velocity":
        plt.plot(t_list, vel_targ, 'g',linewidth=2.0, label="target horizontal speed")
        #plt.plot(t_list, vel_real, 'r',linewidth=1.5, label="measured horizontal speed")
        plt.xlabel("Time (s)",fontsize=20)
        plt.ylabel("Forward velocity of CoM (m/s)", fontsize=20)
        #plt.legend(loc="upper center", fontsize=15)
        #plt.plot(t_list, z_targ, 'g', label="target vertical speed")
        #plt.plot(t_list, s3, 'r', label="measured vertical speed")
        #plt.xlabel("Time (s)", fontsize=20)
        #plt.ylabel("vertical velocity of CoM (m/s)", fontsize=20)
        #plt.legend(loc="lower center", fontsize=20)


        plt.grid()
    if plot_flag=="angle_velocity":
        plt.plot(t_list, s3, 'r')
        #plt.plot(t_list, s5, 'r')
        #plt.plot(t_list, s7, 'r')
        #plt.plot(t_list, s10, 'r')
        #plt.plot(t_list, s12, 'r')
    if plot_flag=="all":
        fig1 = plt.figure(1)
        plt.plot(t_list, gamma_targ_0, 'g', t_list, gamma_real_0, 'r')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Hip angles", fontsize=12)
        fig1.show()
        fig2 = plt.figure(2)
        plt.plot(t_list, knee_targ_0, 'g', t_list, knee_real_0, 'r')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Knee angles", fontsize=12)
        fig2.show()
        fig3 = plt.figure(3)
        plt.plot(t_list, vel_real, 'r', t_list, vel_targ, 'g')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Velocity", fontsize=12)
        fig3.show()
        fig4 = plt.figure(4)
        plt.plot(t_list, hull_real, 'r', t_list, hull_targ, 'g')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("hull angle", fontsize=12)
        fig4.show()
    if plot_flag == "hull angle":
        fig1 = plt.figure(1)
        plt.plot(t_list, hull_real, 'r', t_list, hull_targ, 'g')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("hull angle", fontsize=12)
        fig1.show()

    if plot_flag == "hip filters":
        fig1 = plt.figure(1)
        red, = plt.plot(t_list, hip1_vel, 'r')
        green, = plt.plot(t_list, hip1_vel_filter, 'g')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Hip angular velocity (rad/s)", fontsize=12)
        plt.legend([red, (red, green)], ["Raw", "Filtered"])
        fig1.show()

        fig2 = plt.figure(2)
        red, = plt.plot(t_list, hip2_vel, 'r')
        green, = plt.plot(t_list, hip2_vel_filter, 'g')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Hip2 angular velocity", fontsize=12)
        plt.legend([red, (red, green)], ["Raw", "Filtered"])
        fig2.show()

        fig5 = plt.figure(5)
        plt.plot(t_list, hip_todo_plot, 'r')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Hip to do 1(torque)", fontsize=12)
        fig5.show()

        fig6 = plt.figure(6)
        red, = plt.plot(t_list, s4, 'ro')
        green, = plt.plot(t_list, hip_targ_plot, 'g^')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Hip angle (rad)", fontsize=12)
        plt.legend([red, (red, green)], ["Actual", "Target"])
        fig6.show()

        fig7 = plt.figure(7)
        red, = plt.plot(t_list, s4, 'r')
        green, = plt.plot(t_list, hip1_ang_filter, 'g')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Hip angle (rad)", fontsize=12)
        plt.legend([red, (red, green)], ["Raw", "Filtered"])
        fig7.show()
    if plot_flag == "knee filters":
        fig1 = plt.figure(1)
        red, = plt.plot(t_list, knee1_vel, 'r')
        green, = plt.plot(t_list, knee1_vel_filter, 'g')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Knee1 angular velocity", fontsize=12)
        plt.legend([red, (red, green)], ["Raw", "Filtered"])
        fig1.show()

        fig2 = plt.figure(2)
        red, = plt.plot(t_list, knee2_vel, 'r')
        green, = plt.plot(t_list, knee2_vel_filter, 'g')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Knee2 angular velocity", fontsize=12)
        plt.legend([red, (red, green)], ["Raw", "Filtered"])
        fig2.show()

        fig6 = plt.figure(6)
        red, = plt.plot(t_list, s6, 'ro')
        green, = plt.plot(t_list, hip_targ_plot, 'g^')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("knee angle", fontsize=12)
        plt.legend([red, (red, green)], ["Actual", "Target"])
        fig6.show()
    #vel_filter = []
    #alpha = 10.0
    #for j in xrange(len(s3)):
    #    vel = s3[j]
    #    height = height_real[j]
    #    s = vel/(h_c-height)
    #    vel_filter.append(vel/(alpha/(alpha+s)))
    #plt.plot(t_list, vel_filter)

    plt.show()