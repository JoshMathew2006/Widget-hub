import pymunk

WORLD_WIDTH = 800
WORLD_HEIGHT = 600

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

    def add_box(self, obj_id, x, y, window_type="box", content=None, width=20, height=20):
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


    def step(self, dt):
        substeps = 4
        for _ in range(substeps):
            self.space.step(dt/substeps)

    def resize_box(self, obj_id, factor):
        if obj_id in self.sizes:
            old_width, old_height = self.sizes[obj_id]
            new_width = int(old_width * factor)
            new_height = int(old_height * factor)

            min_width, min_height = 200, 150  # default minimum
            if self.types.get(obj_id) == "chat":
                min_width, min_height = 280, 250  # chat needs more room

            new_width = max(new_width, min_width)
            new_height = max(new_height, min_height)

            self.sizes[obj_id] = (new_width, new_height)

            body = self.bodies[obj_id]
            for shape in list(body.shapes):
                if shape in self.space.shapes:
                    self.space.remove(shape)

            half_w = new_width / 2
            half_h = new_height / 2
            new_shape = pymunk.Poly(body, [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
            self.space.add(new_shape)

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
    
    def get_body_at(self, x , y):
        found = self.space.point_query_nearest((x,y) , 10, pymunk.ShapeFilter())
        print(f"query result: {found}")
        if found is None:
            return None
        target_body = found.shape.body
        for obj_id, body in self.bodies.items():
            if body is target_body:
                return obj_id
        return None

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