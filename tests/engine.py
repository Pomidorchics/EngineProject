import time


class Ray:
    def __init__(self, cs, initialpt, direction):
        self.cs = cs
        self.initialpt = initialpt
        self.direction = direction


class Identifier:
    def __init__(self):
        pass


def get_id():
    return time.time()


class Entity:
    def __init__(self, cs):
        self.cs = cs
        self.properties = dict()
        self.id = get_id()

    def set_property(self, prop, val):
        self.properties[prop] = val

    def get_property(self, prop):
        return self.properties[prop]

    def remove_property(self, prop):
        del self.properties[prop]


class EntitiesList:
    def __init__(self, entities):
        self.entities = entities

    def append(self, entity):
        self.entities.append(entity)

    def remove(self, entity):
        self.entities.remove(entity)

    def get(self, id):
        pass

    def exec(self, func):
        pass


class Game:
    def __init__(self, cs, entities):
        self.cs = cs
        self.entities = entities

    def run(self):
        pass

    def update(self):
        pass

    def exit(self):
        pass

    def get_entity_class(self):
        pass

    def get_ray_class(self):
        pass


class Object(Entity):
    def __init__(self, position, direction):
        pass

    def move(self):
        pass

    def planar_rotate(self):
        pass

    def rotate_3d(self):
        pass

    def set_position(self):
        pass

    def set_direction(self):
        pass


class Camera(Object):
    pass
