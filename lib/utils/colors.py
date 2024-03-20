class RGBA:
    def __init__(self, red, green, blue, alpha=255):
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def __str__(self):
        return "({},{},{},{})".format(self.red, self.green, self.blue, self.alpha)

    @property
    def hex(self):
        return "#{:02X}{:02X}{:02X}".format(self.red, self.green, self.blue)

    @property
    def rgba(self, alpha=255):
        return (self.red, self.green, self.blue, alpha)

    @property
    def rgb(self):
        return (self.red, self.green, self.blue)

    @property
    def bgra(self, alpha=255):
        return (self.blue, self.green, self.red, alpha)

    @property
    def bgr(self):
        return (self.blue, self.green, self.red)

    @property
    def rgba_norm(self, alpha=255):
        return (self.red / 255.0, self.green / 255.0, self.blue / 255.0, alpha / 255.0)

    @property
    def rgb_norm(self):
        return (self.red / 255.0, self.green / 255.0, self.blue / 255.0)

    @property
    def bgr_norm(self):
        return (self.blue / 255.0, self.green / 255.0, self.red / 255.0)

    @property
    def bgra_norm(self, alpha=255):
        return (self.blue / 255.0, self.green / 255.0, self.red / 255.0, alpha / 255.0)


COLORS = {
    "red": RGBA(255, 0, 0, 255),
    "dark_red": RGBA(128, 0, 0, 255),
    "green": RGBA(0, 255, 0, 255),
    "dark_green": RGBA(0, 128, 0, 255),
    "blue": RGBA(0, 0, 255, 255),
    "yellow": RGBA(255, 255, 0, 255),
    "magenta": RGBA(255, 0, 255, 255),
    "cyan": RGBA(0, 255, 255, 255),
    "orange": RGBA(255, 165, 0, 255),
    "purple": RGBA(128, 0, 128, 255),
    "brown": RGBA(165, 42, 42, 255),
    "pink": RGBA(255, 192, 203, 255),
    "lime": RGBA(0, 255, 127, 255),
    "navy": RGBA(0, 0, 128, 255),
    "teal": RGBA(0, 128, 128, 255),
    "olive": RGBA(128, 128, 0, 255),
    "maroon": RGBA(128, 0, 0, 255),
    "coral": RGBA(255, 127, 80, 255),
    "turquoise": RGBA(64, 224, 208, 255),
    "indigo": RGBA(75, 0, 130, 255),
    "violet": RGBA(238, 130, 238, 255),
    "gold": RGBA(255, 215, 0, 255),
    "skin": RGBA(192, 134, 107, 255),
    "white": RGBA(255, 255, 255, 255),
    "black": RGBA(0, 0, 0, 255),
    "gray": RGBA(128, 128, 128, 255),
    "darkgray": RGBA(64, 64, 64, 255),
    "lightgray": RGBA(192, 192, 192, 255),
    "tomato": RGBA(255, 99, 71, 255),
}


# RGB colors for Object classes
OBJ_CLASS_COLORS = [
    COLORS["black"],  # background
    COLORS["red"],  # object_id 1
    COLORS["green"],  # object_id 2
    COLORS["blue"],  # object_id 3
    COLORS["yellow"],  # object_id 4
]

OBJECT_COLORS = OBJ_CLASS_COLORS


# RGB colors for Hands
HAND_COLORS = [
    COLORS["black"],  # background
    COLORS["turquoise"],  # right
    COLORS["tomato"],  # left
]

HAND_BONE_COLORS = [
    # palm
    COLORS["gray"],  # (0, 1)
    COLORS["gray"],  # (0, 5)
    COLORS["gray"],  # (0, 17)
    COLORS["gray"],  # (5, 9)
    COLORS["gray"],  # (9, 13)
    COLORS["gray"],  # (13, 17)
    # thumb
    COLORS["red"],  # (1, 2)
    COLORS["red"],  # (2, 3)
    COLORS["red"],  # (3, 4)
    # index
    COLORS["green"],  # (5, 6)
    COLORS["green"],  # (6, 7)
    COLORS["green"],  # (7, 8)
    # middle
    COLORS["blue"],  # (9, 10)
    COLORS["blue"],  # (10, 11)
    COLORS["blue"],  # (11, 12)
    # ring
    COLORS["yellow"],  # (13, 14)
    COLORS["yellow"],  # (14, 15)
    COLORS["yellow"],  # (15, 16)
    # pinky
    COLORS["pink"],  # (17, 18)
    COLORS["pink"],  # (18, 19)
    COLORS["pink"],  # (19, 20)
]

HAND_JOINT_COLORS = [
    # root
    COLORS["black"],  # 0
    # thumb
    COLORS["red"],  # 1
    COLORS["red"],  # 2
    COLORS["red"],  # 3
    COLORS["red"],  # 4
    # index
    COLORS["green"],  # 5
    COLORS["green"],  # 6
    COLORS["green"],  # 7
    COLORS["green"],  # 8
    # middle
    COLORS["blue"],  # 9
    COLORS["blue"],  # 10
    COLORS["blue"],  # 11
    COLORS["blue"],  # 12
    # ring
    COLORS["yellow"],  # 13
    COLORS["yellow"],  # 14
    COLORS["yellow"],  # 15
    COLORS["yellow"],  # 16
    # pinky
    COLORS["pink"],  # 17
    COLORS["pink"],  # 18
    COLORS["pink"],  # 19
    COLORS["pink"],  # 20
]
