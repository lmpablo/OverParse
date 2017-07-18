import cv2
import numpy as np
import os
import csv
import argparse
import json
import pprint
from templates import heroes, win_state, TEMPLATE_DIR
from matplotlib import pyplot as plt


class MatchResults(object):
    def __init__(self, win_states, player_heroes, opponent_heroes):
        rounds = []
        for round_win, p_hero, o_hero in zip(win_states, player_heroes, opponent_heroes):
            print round_win
            rounds.append({
                'player_win': round_win[0] == 'PLAYER_WON',
                'player_hero': p_hero,
                'opponent_hero': o_hero,
                'opponent_win': round_win[0] == 'PLAYER_LOST'
            })
        self.data = rounds

    def to_json(self):
        return self.data

    def to_csv(self, filename):
        headers = self.data[0].keys() if self.data else []
        with open(filename, 'wb') as out:
            csv_writer = csv.DictWriter(out, fieldnames=headers)
            csv_writer.writeheader()
            csv_writer.writerows(self.data)


class Template(object):
    def __init__(self, name, template_src):
        self.name = name
        self.template_dir = TEMPLATE_DIR
        self.template = cv2.imread(os.path.join(TEMPLATE_DIR, template_src), 0)
        w, h = self.template.shape[::-1]
        self.width = w
        self.height = h


class ScreenshotScanner(object):
    def __init__(self, method=cv2.TM_CCORR_NORMED):
        self.heroes = heroes
        self.hero_templates = {}
        self.win_state_templates = {}

        self.load_win_state_templates()
        self.load_hero_templates()
        self.method = method


    def load_hero_templates(self):
        for hero in self.heroes:
            self.hero_templates[hero['name']] = Template(hero['name'], hero['template_src'])

    def load_win_state_templates(self):
        for item in win_state:
            self.win_state_templates[item['name']] = Template(item['name'], item['template_src'])

    def merge_points(self, points, threshold):
        merged = set()

        previous = None
        for pt in points:
            if previous is not None:
                if not self._can_merge(previous, pt, threshold):
                    if not any([self._can_merge(item, pt, threshold) for item in merged]):
                        merged.add(pt)
            else:
                merged.add(pt)
            previous = pt

        return merged

    def _can_merge(self, pair1, pair2, threshold=5):
        x1, y1 = pair1
        x2, y2 = pair2

        return (x1 == x2 and y1 == y2) or (abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold)

    def _generate_debug(self, points, debug_image, label):
        for pt in points:
            cv2.putText(debug_image, label, pt, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

            cv2.imwrite('debug.png', debug_image)

    def _extract_data(self, input_data, template_group, confidence_threshold=0.95, with_debug=None):
        extracted = {}
        for title, data in template_group.iteritems():
            res = cv2.matchTemplate(input_data, data.template, self.method)
            loc = np.where(res >= confidence_threshold)

            points = self.merge_points(zip(*loc[::-1]), min(data.width, data.height))
            extracted[title] = points

            if with_debug is not None:
                self._generate_debug(points, with_debug, title)

        return extracted

    def _process_win_state_data_points(self, data_points):
        combined_win_states = []
        for key, values in data_points.iteritems():
            combined_win_states += [(key, v) for v in values]

        return sorted(combined_win_states, key=lambda d: d[1][0])

    def _process_hero_data_points(self, data_points, midpoint):
        player = {}
        opponent = {}

        for hero, dp in data_points.iteritems():
            for point in dp:
                x, y = point
                if y > midpoint:
                    player[point] = hero
                else:
                    opponent[point] = hero

        p_points = [h[1] for h in sorted(player.iteritems(), key=lambda d: d[0])]
        o_points = [h[1] for h in sorted(opponent.iteritems(), key=lambda d: d[0])]
        # p_points = sorted([hero for point, hero in player.iteritems()], key=lambda d: player[d])
        # o_points = sorted([hero for point, hero in opponent.iteritems()], key=lambda d: opponent[d])

        print(p_points)
        return p_points, o_points


    def process(self, input_src, confidence_threshold=0.95, with_debug=False):
        img_rgb = cv2.imread(input_src)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        debug_image = cv2.imread(input_src)

        win_state_data_points = self._extract_data(img_gray,
                                            self.win_state_templates,
                                            confidence_threshold,
                                            debug_image if with_debug else None)

        sorted_win_states = self._process_win_state_data_points(win_state_data_points)

        if len(sorted_win_states) == 0:
            raise RuntimeError("No win states found")

        midpoint = sorted_win_states[0][1][1]
        hero_data_points = self._extract_data(img_gray,
                                              self.hero_templates,
                                              confidence_threshold,
                                              debug_image if with_debug else None)

        sorted_player_points, sorted_opponent_points = self._process_hero_data_points(hero_data_points, midpoint)

        return MatchResults(sorted_win_states, sorted_player_points, sorted_opponent_points)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('input', help='The input file to process')
    argparser.add_argument('--debug', action='store_true', help='Produce a debug image output')
    argparser.add_argument('--confidence', type=float, default=0.95)
    argparser.add_argument('--output', help='Optional output filename')
    argparser.add_argument('--output-type', choices=['json', 'csv'])

    args = argparser.parse_args()

    ss = ScreenshotScanner()
    results = ss.process(args.input, confidence_threshold=args.confidence, with_debug=args.debug)

    if args.output:
        if args.output_type == 'json':
            with open(args.output, 'w') as outfile:
                outfile.write(json.dumps(results.to_json()))
        elif args.output_type == 'csv':
            results.to_csv(args.output)
    else:
        pprint.pprint(json.dumps(results.to_json()))



if __name__ == '__main__':
    main()