"""MediaPipe Hands connections."""

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17))
HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))


HAND_BONES = (
    HAND_PALM_CONNECTIONS
    + HAND_THUMB_CONNECTIONS
    + HAND_INDEX_FINGER_CONNECTIONS
    + HAND_MIDDLE_FINGER_CONNECTIONS
    + HAND_RING_FINGER_CONNECTIONS
    + HAND_PINKY_FINGER_CONNECTIONS
)

HAND_JOINT_NAMES = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

HAND_JOINT_PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    0,
    13,
    14,
    15,
    0,
    17,
    18,
    19,
]

# add faces that make the hand mesh watertight
NEW_MANO_FACES = [
    [92, 38, 234],
    [234, 38, 239],
    [38, 122, 239],
    [239, 122, 279],
    [122, 118, 279],
    [279, 118, 215],
    [118, 117, 215],
    [215, 117, 214],
    [117, 119, 214],
    [214, 119, 121],
    [119, 120, 121],
    [121, 120, 78],
    [120, 108, 78],
    [78, 108, 79],
]
