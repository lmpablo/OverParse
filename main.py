import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_positions(input_image_src, template_image_src, debug=None, text=''):
    img_rgb = cv2.imread(input_image_src)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_image_src, 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
    threshold = 0.95
    loc = np.where(res >= threshold)

    s = set()
    previous = None
    for pt in zip(*loc[::-1]):
        if previous is not None:
            if not can_merge(previous, pt, min(w, h)):
                if not any([can_merge(item, pt, min(w, h)) for item in s]):
                    s.add(pt)
        else:
            s.add(pt)
        previous = pt

    if debug is not None:
        for pt in s:
            cv2.putText(img_rgb, text, pt, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        cv2.imwrite('{}.png'.format(debug), img_rgb)

    return s


def can_merge(pair1, pair2, threshold=5):
    x1, y1 = pair1
    x2, y2 = pair2

    if x1 == x2 and y1 == y2:
        return True
    elif abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold:
        return True
    else:
        return False

PLAYER_WON = 'templates/player_won.jpg'
PLAYER_LOST = 'templates/player_lost.jpg'
INPUT_IMAGE = 'winscreen.jpg'

# scan for win and loss markers
wins = get_positions(INPUT_IMAGE, PLAYER_WON, 'res', text='won')
losses = get_positions('res.png', PLAYER_LOST, 'res', text='lost')

# combine the result markers and sort
all_stats = wins.union(losses)
sorted_by_position = sorted(all_stats, key=lambda s: s[0])
rounds = [(s in wins) for s in sorted_by_position]

# use the y value as the marker whether
# a character scanned is by the player or their opponent
midpoint = sorted_by_position[0][1]

player_rounds = {}
opp_rounds = {}

# scan for every hero
for character in ['ana', 'genji', 'hanzo', 'junkrat',
                  'mccree', 'mei', 'pharah', 'reaper',
                  'roadhog', 'soldier76', 'sombra', 'tracer',
                  'widowmaker', 'zenyatta']:
    positions = get_positions('res.png', 'templates/{}.jpg'.format(character), None)
    if positions:
        for p in positions:
            if p[1] > midpoint:
                player_rounds[p] = character
            else:
                opp_rounds[p] = character

player_characters = [player_rounds[c] for c in sorted(player_rounds, key=lambda pr: pr[0])]
opp_characters = [opp_rounds[c] for c in sorted(opp_rounds, key=lambda pr: pr[0])]

print "Round,Luis Hero,Jaime Hero,Winner"
for index, result in enumerate(zip(rounds, player_characters, opp_characters)):
    did_win, player, opponent = result
    print ','.join([str(index + 1), player.title(), opponent.title(), 'Luis' if did_win else 'Jaime'])
