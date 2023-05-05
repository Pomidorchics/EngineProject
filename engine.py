import time
from classes import *


class Ray:
    def __init__(self, cs, initial_pt, direction):
        self.cs = cs
        self.initial_pt = initial_pt
        self.direction = direction


class Identifier:
    def __init__(self):
        self.value = time.time()

    @staticmethod
    def generate_id():
        return time.time()


class Entity:
    def __init__(self, cs):
        self.cs = cs
        self.properties = dict()
        self.id = Identifier.generate_id()

    def set_property(self, prop, new_val):
        self.properties[prop] = new_val

    def get_property(self, prop):
        return self.properties[prop]

    def remove_property(self, prop):
        del self.properties[prop]

    def __getitem__(self, prop):
        return self.get_property(prop)

    def __setitem__(self, prop, new_val):
        return self.set_property(prop, new_val)


class EntitiesList:
    def __init__(self, entities):
        self.entities = entities

    def append(self, entity):
        self.entities.append(entity)

    def remove(self, entity):
        self.entities.remove(entity)

    def get(self, id):
        for entity in self.entities:
            if entity.id == id:
                return entity

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
    def __init__(self, fov, draw_distance, v_fov=0, look_at=Point([0, 0, 0])):
        self.fov = fov
        self.draw_distance = draw_distance
        self.v_fov = v_fov
        self.look_at = look_at

