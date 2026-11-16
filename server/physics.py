import pymunk

WORLD_WIDTH = 800
WORLD_HEIGHT = 600

# type: (min_w, min_h, max_w, max_h, keep_aspect)   sizes are in world units (800x600)
SIZE_LIMITS = {
    "box":     (60,  60,  300, 300, True),
    "note":    (120, 150, 500, 500, False),
    "chat":    (160, 220, 450, 580, False),
    "weather": (200, 111, 540, 300, True),   # same 1.8 shape all the way down
    "spotify": (100, 137, 300, 410, True),
}
DEFAULT_LIMITS = (60, 60, 400, 400, False)

class PhysicsWorld:
    def __init__(self):
        self.space = pymunk.Space()
        self.bodies = {}
        self.sizes = {}
        self.types = {}
        self.contents = {}
        self.setup_ground()
        self.setup_walls()
        self.setup_ceiling()


    def setup_ground(self):
        ground_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(ground_body, (0, WORLD_HEIGHT), (WORLD_WIDTH, WORLD_HEIGHT), 20)
        ground_shape.friction = 1.0
        self.space.add(ground_body, ground_shape)
        pass

    def setup_walls(self):
        wall1_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        wall1_shape = pymunk.Segment(wall1_body, (0,0) , (0, WORLD_HEIGHT), 20)
        self.space.add(wall1_body, wall1_shape)
        wall2_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        wall2_shape = pymunk.Segment(wall2_body, (WORLD_WIDTH, 0), (WORLD_WIDTH, WORLD_HEIGHT), 20)
        self.space.add(wall2_body, wall2_shape)

    def setup_ceiling(self):
        ceiling_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        ceiling_shape = pymunk.Segment(ceiling_body, (0,0), (WORLD_WIDTH, 0), 20)
        self.space.add(ceiling_body, ceiling_shape)

    def add_box(self, obj_id, x, y, window_type="box", content=None, width=150, height= 150):
        new_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        new_body.position = (x, y)
        hw, hh = width / 2, height / 2
        new_body_shape = pymunk.Poly(new_body, [(-hw,-hh), (hw,-hh), (hw,hh), (-hw,hh)])
        self.space.add(new_body, new_body_shape)
        self.bodies[obj_id] = new_body
        self.sizes[obj_id] = (width, height)
        self.types[obj_id] = window_type
        self.contents[obj_id] = content

    def set_content(self, obj_id, content):
        if obj_id in self.contents:
            self.contents[obj_id] = content

    def set_position(self, obj_id, x, y):
        if obj_id in self.bodies:
            size = self.sizes.get(obj_id, (20,20))
            half_w , half_h = size[0] / 2, size[1] / 2
            clamped_x = max(half_w, min(WORLD_WIDTH - half_w, x))
            clamped_y = max(half_h, min(WORLD_HEIGHT - half_h, y))
            self.bodies[obj_id].position = (clamped_x, clamped_y)

    def bring_to_front(self, obj_id):
        if obj_id in self.bodies:
            self.bodies[obj_id] = self.bodies.pop(obj_id)

    def step(self, dt):
        substeps = 4
        for _ in range(substeps):
            self.space.step(dt/substeps)


    def _rebuild_shape(self, obj_id):
        body = self.bodies[obj_id]
        for shape in list(body.shapes):
            if shape in self.space.shapes:
                self.space.remove(shape)
        hw, hh = self.sizes[obj_id][0] / 2, self.sizes[obj_id][1] / 2
        self.space.add(pymunk.Poly(body, [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]))

    def set_size(self, obj_id, w, h, keep_top_left=True):
        # every resize goes through here, so the limits always apply
        if obj_id not in self.sizes:
            return
        min_w, min_h, max_w, max_h, keep_aspect = SIZE_LIMITS.get(self.types[obj_id], DEFAULT_LIMITS)
        old_w, old_h = self.sizes[obj_id]
        if keep_aspect:
            s = max(w / old_w, h / old_h)                         # follow whichever side grew more
            s_min = max(min_w / old_w, min_h / old_h)
            s_max = min(max_w / old_w, max_h / old_h)
            s = max(s_min, min(s_max, s))
            w, h = old_w * s, old_h * s
        else:
            w = max(min_w, min(max_w, w))
            h = max(min_h, min(max_h, h))

        x, y = self.bodies[obj_id].position
        self.sizes[obj_id] = (w, h)
        self._rebuild_shape(obj_id)
        if keep_top_left:   # corner drags: top-left stays put
            self.set_position(obj_id, x - old_w / 2 + w / 2, y - old_h / 2 + h / 2)
        else:               # button/factor resizes: grow from the center
            self.set_position(obj_id, x, y)   # still re-clamps it inside the walls

    def resize_box(self, obj_id, factor):
        if obj_id in self.sizes:
            w, h = self.sizes[obj_id]
            self.set_size(obj_id, w * factor, h * factor, keep_top_left=False)
    
    def get_state(self):

        stateDict = {}
        for obj_id, body in self.bodies.items():
            stateDict[obj_id] = {
                "x": body.position.x,
                "y": body.position.y,
                "angle": body.angle,
                "size": self.sizes[obj_id],
                "type": self.types[obj_id],
                "content": self.contents[obj_id]
            }

        return stateDict
    
    def get_body_at(self, x, y, reach=30):
        hits = []
        for q in self.space.point_query((x, y), reach, pymunk.ShapeFilter()):
            for order, (obj_id, body) in enumerate(self.bodies.items()):
                if body is q.shape.body:
                    inside = q.distance <= 0
                    is_overlay = self.types[obj_id] != "box"
                    hits.append(((inside, is_overlay, order), obj_id))
        if not hits:
            return None
        return max(hits)[1]

    def remove_box(self, obj_id):
        if obj_id in self.bodies:
            body = self.bodies[obj_id]

            for shape in list(body.shapes):
                if shape in self.space.shapes:
                    self.space.remove(shape)

            self.space.remove(body)
            del self.bodies[obj_id]
            del self.sizes[obj_id]
            del self.types[obj_id]
            del self.contents[obj_id]
    
    def append_chat_message(self, obj_id, role, text):
        if obj_id in self.contents:
            self.contents[obj_id].append({"role": role, "text": text})