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
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 100
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 1000
SPEED_HIP     = 12
SPEED_KNEE    = 15
LIDAR_RANGE   = 160/SCALE

HULL_POLY =[
    (-30,+10), (+30,+10),
    (+30,-10), (-30,-10),
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
T_step = 0.4    # step time (s)
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

        self.world = Box2D.b2World(gravity=(0,-10), doSleep=True)
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

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        self.hull = self.world.CreateDynamicBody(
            position = (hull_x, (hull_y-h_c/1.2)),
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
                localAnchorB=(0, -LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 0.0,
                referenceAngle = 0.0,
                lowerAngle = -0.0001,#-1.4,      # -80 degree
                upperAngle = 0.00001,#1.4,       # 80 degree
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
                localAnchorA=(0, LEG_H/2),
                localAnchorB=(0, -LEG_H/2),  
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 0.0,
                referenceAngle = 0.0,#1*math.pi,
                lowerAngle = 0.0,#0.35,#3.1399,    # 20 degree
                upperAngle = 0.001,#3.14,#3.1401,    # 180 degree
                )

            # Add a foot
            foot = self.world.CreateDynamicBody(
                position = (init_x, init_y + h_c/1.2),
                # angle = (-math.pi/2+a_i),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/3)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            foot.color1 = (1.0-i/10., 1.0-i/10., 1.0-i/10.)
            foot.color2 = (0.1-i/10., 0.1-i/10., 0.1-i/10.)

            foot_jd = revoluteJointDef(
                bodyA=lower,
                bodyB=foot,
                localAnchorA=(0, LEG_H/2),
                localAnchorB=(0, LEG_H/3),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 0.0,
                referenceAngle = 1.4,
                lowerAngle = -1.4,
                upperAngle = 1.4,
                )

            lower.ground_contact = False
            foot.ground_contact = False

            self.legs.append(lower)
            self.legs.append(foot)
            self.joints.append(self.world.CreateJoint(rjd))
            self.joints.append(self.world.CreateJoint(foot_jd))

        self.drawlist = self.terrain + self.legs + [self.hull]

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
            self.joints[0].angle,       #s[4] hip 1
            self.joints[0].speed,       #s[5]
            self.joints[1].angle,       #s[6] knee 1
            self.joints[1].speed,       #s[7]
            self.joints[2].angle,       #s[8] ankle 1
            self.joints[2].speed,       #s[9]
            1.0 if self.legs[2].ground_contact else 0.0,    #s[10]   
            self.joints[3].angle,                           #s[11] hip 2
            self.joints[3].speed,                           #s[12]
            self.joints[4].angle,                           #s[13] knee 2
            self.joints[4].speed,                           #s[14]
            self.joints[5].angle,                           #s[15] ankle 2
            self.joints[5].speed,                           #s[16]
            1.0 if self.legs[4].ground_contact else 0.0,     #s[17]  
            pos.y - TERRAIN_HEIGHT,                          #s[18]
            pos.x - self.legs[1].position.x,    # s[19]
            pos.x - self.legs[3].position.x,     # s[20]
            pos.x - init_x,   # global xcom      s[21]
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state)==32

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
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

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

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

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
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    RISING, FALLING = 1, 2
    stage = RISING
    moving_leg = 0
    supporting_leg = 1 - moving_leg

    T_m = T_step / 2        # the moving leg will reach the highest point of its trajectory at half the step time

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
    torque_a0, torque_a1, torque_a2, torque_a3, torque_a4, torque_a5 = [], [], [],[],[],[]
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

    # For posture control,  To be accumulated during stance
    delta_hip_angle = 0.0

    # Initialisation for filter
    x_ft = np.zeros((7,2))
    raw_data = np.zeros((7,2))
    ankle_ang_real, ankle_desired, ankle_vel, ankle_vel_filter = [], [], [], []

    while True:
        t = (steps)/float(FPS)

        s, r, done, info = env.step(a)
        steps += 1

        ankle_targ = [0.0, 0.0]
        ankle_todo = [0.0, 0.0]
        # Desired angle for tuning (sine wave)
        desired_angle = math.sin(steps/FPS*2)
        
        ankle_targ[0] = desired_angle
        ankle_targ[1] = desired_angle

        ankle_vel.append(s[9])
        ankle_desired.append(ankle_targ[1])
        # Create the raw data matrix
        raw_data = [[s[4], s[5]],   # Hip1
                    [s[6], s[7]],   # Knee1
                    [s[8], s[9]],   # Ankle1
                    [s[11], s[12]], # Hip2
                    [s[13], s[14]], # Knee2
                    [s[15], s[16]], # Ankle2
                    [s[0], s[1]]]   # hull

        # Filter raw data
        x_ft += 7.5*(raw_data - x_ft) * 1.0/FPS

        ankle_vel_filter.append(x_ft[2][1])

        ankle_todo[0] = 20.0 * (ankle_targ[0] - s[8]) - 1.2 * x_ft[2][1]
        ankle_todo[1] = 20.0 * (ankle_targ[1] - s[15]) - 1.2 * x_ft[5][1]

        hull_real.append(s[0])   # body angle
        torque_a0.append(a[0])
        torque_a1.append(a[1])
        torque_a2.append(a[2])
        torque_a3.append(a[3])
        torque_a4.append(a[4])
        torque_a5.append(a[5])

        # Ankle angles graphs
        ankle_ang_real.append(s[8])

        a[0] = 0.0
        a[1] = 0.0
        a[2] = ankle_todo[0]
        a[3] = 0.0
        a[4] = 0.0
        a[5] = ankle_todo[1]

        env.render()

        if steps >= FPS*10: break
        # if done: break

    z_targ = []

    # plot
    t_list = []
    # print("length of gamma targ 0: " + str(len(gamma_targ_0)))
    for j in range(len(ankle_desired)):
        t_list.append(float(j+1)/float(FPS))
        height_targ.append(h_c/2)
        #hull_targ.append(0)
        z_targ.append(0)

    fig6 = plt.figure(6)
    red, = plt.plot(t_list, ankle_ang_real, 'ro')
    green, = plt.plot(t_list, ankle_desired, 'g^')
    plt.xlabel("Time (s)", fontsize=10)
    plt.ylabel("Ankle angle (rad)", fontsize=10)
    plt.legend([red, (red, green)], ["Actual", "Target"])
    fig6.show()

    fig1 = plt.figure(1)
    red, = plt.plot(t_list, ankle_vel, 'r')
    green, = plt.plot(t_list, ankle_vel_filter, 'g')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Ankle angular velocity (rad/s)", fontsize=12)
    plt.legend([red, (red, green)], ["Raw", "Filtered"])
    fig1.show()
    #vel_filter = []
    #alpha = 10.0
    #for j in xrange(len(s3)):
    #    vel = s3[j]
    #    height = height_real[j]
    #    s = vel/(h_c-height)
    #    vel_filter.append(vel/(alpha/(alpha+s)))
    #plt.plot(t_list, vel_filter)

    plt.show()