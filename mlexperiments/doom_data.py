import random
import json
import os
from threading import Thread
import numpy as np
import vizdoom as vz
from oblige import DoomLevelGenerator
from skimage import transform
from torch.utils.data import Dataset
from PIL import Image
from PIL.PngImagePlugin import PngImageFile, PngInfo


ACTOR_NAMES = [
    "DoomPlayer", "Marine", "ZombieMan", "ShotgunGuy", "Archvile", "ArchvileFire", "Revenant", "RevenantTracer",
    "RevenantTracerSmoke", "Fatso", "FatShot", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "Cacodemon",
    "BaronOfHell", "BaronBall", "HellKnight", "LostSoul", "SpiderMastermind", "Arachnotron", "Cyberdemon",
    "PainElemental", "WolfensteinSS", "CommanderKeen", "BossBrain", "BossEye", "BossTarget", "SpawnShot",
    "SpawnFire", "ExplosiveBarrel", "DoomImpBall", "CacodemonBall", "Rocket", "PlasmaBall", "BFGBall",
    "ArachnotronPlasma", "BulletPuff", "Blood", "TeleportFog", "ItemFog", "TeleportDest", "BFGExtra",
    "GreenArmor", "BlueArmor", "HealthBonus", "ArmorBonus", "BlueCard", "RedCard", "YellowCard",
    "YellowSkull", "RedSkull", "BlueSkull", "Stimpack", "Medikit", "Soulsphere", "InvulnerabilitySphere",
    "Berserk", "BlurSphere", "RadSuit", "Allmap", "Infrared", "Megasphere", "Clip", "ClipBox", "RocketAmmo",
    "RocketBox", "Cell", "CellPack", "Shell", "ShellBox", "Backpack", "BFG9000", "Chaingun", "Chainsaw",
    "RocketLauncher", "PlasmaRifle", "Shotgun", "SuperShotgun"
]
BLOCKING_ACTOR_NAMES = [
    "TechLamp", "TechLamp2", "Column", "TallGreenColumn", "ShortGreenColumn", "TallRedColumn",
    "ShortRedColumn", "SkullColumn", "HeartColumn", "EvilEye", "FloatingSkull", "TorchTree", "BlueTorch",
    "GreenTorch", "RedTorch", "ShortBlueTorch", "ShortGreenTorch", "ShortRedTorch", "Stalagtite",
    "TechPillar", "CandleStick", "Candelabra", "BloodyTwitch", "Meat2", "Meat3", "Meat4", "Meat5",
    "GibbedMarine", "GibbedMarineExtra", "HeadsOnAStick", "Gibs", "HeadOnAStick", "HeadCandles",
    "DeadStick", "LiveStick", "BigTree", "BurningBarrel", "HangNoGuts", "HangBNoBrain", "HangTLookingDown",
    "HangTSkull", "HangTLookingUp", "HangTNoBrain", "ColonGibs", "SmallBloodPool", "BrainStem", "RealGibs"
]
ACTOR_IDS = {ACTOR_NAMES[i].upper(): i + 1 for i in range(len(ACTOR_NAMES))}
ACTOR_IDS.update({actor_name.upper(): max([ACTOR_IDS[x] for x in ACTOR_IDS]) + 1 for actor_name in BLOCKING_ACTOR_NAMES})
ACTOR_IDS.update({"DEAD" + actor_name: max([ACTOR_IDS[x] for x in ACTOR_IDS]) + 1 for actor_name in ACTOR_IDS})

def labels_to_types(labels, labels_buffer):
    types_buffer = np.zeros_like(labels_buffer, dtype=int)
    for label in labels:
        actor_name = label.object_name.upper()
        if actor_name in ACTOR_IDS:
            types_buffer[labels_buffer == label.value] = ACTOR_IDS[actor_name]
        else:
            print("Unhandled actor type: ", actor_name)
    return types_buffer

def labels_to_instances(labels, labels_buffer):
    instances_buffer = np.zeros_like(labels_buffer, dtype=int)
    for label in labels:
        instances_buffer[labels_buffer == label.value] = (label.object_id % 255) + 1
    return instances_buffer


class DoomWadGenerator:
    def __init__(self, id=0, config={"length": "single"}, seed=None):
        super(DoomWadGenerator, self).__init__()
        self.id = id
        self.config = config
        self.seed = seed
        
    def __call__(self):
        generator = DoomLevelGenerator(self.seed)
        generator.set_config(self.config)
        seed = generator.get_seed()
        wad_fname = "oblige{}.wad".format(self.id)
        num_maps = generator.generate(wad_fname)
        sample = {"seed": seed, "wad_fname": wad_fname, "num_maps": num_maps}
        return sample

class DoomMapGenerator:
    def __init__(self, wad_generator=None):
        self.wad_generator = wad_generator or DoomWadGenerator()
        self.wad = None
        self.map_idx = 0

    def __call__(self):
        while True:
            if not self.wad:
                self.wad = self.wad_generator()
                self.map_idx = 0
            if self.map_idx >= self.wad["num_maps"]:
                self.wad = None
            else:
                break
        self.map_idx += 1
        sample = {"seed": self.wad["seed"], "wad_fname": self.wad["wad_fname"], "map_name": "map{:02d}".format(self.map_idx)}
        return sample

class DoomGameGenerator:
    BUTTONS = [vz.Button.MOVE_FORWARD, vz.Button.MOVE_BACKWARD, vz.Button.MOVE_LEFT, vz.Button.MOVE_RIGHT, vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT, vz.Button.ATTACK, vz.Button.USE]
    VARIABLES = [vz.GameVariable.POSITION_X, vz.GameVariable.POSITION_Y, vz.GameVariable.POSITION_Z, vz.GameVariable.ANGLE]
    
    def __init__(self, config_file, map_generator=None, resolution=vz.ScreenResolution.RES_320X240, format=vz.ScreenFormat.RGB24, buttons=BUTTONS, depth=True, labels=True):
        super(DoomGameGenerator, self).__init__()
        self.config_file = config_file
        self.map_generator = map_generator or DoomMapGenerator()
        self.resolution = resolution
        self.format = format
        self.buttons = buttons
        self.depth = depth
        self.labels = labels

    def __call__(self):
        game = vz.DoomGame()
        game.load_config(self.config_file)
        game.add_game_args("+gamma 2")

        game.set_screen_resolution(self.resolution)
        game.set_screen_format(self.format)
        game.set_depth_buffer_enabled(self.depth)
        game.set_labels_buffer_enabled(self.labels)

        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(False)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)
        game.set_render_corpses(False)
        game.set_window_visible(False)

        game.set_available_buttons(self.buttons)
        game.set_available_game_variables(DoomGameGenerator.VARIABLES)
        game.set_episode_timeout(2**31 - 1)
    
        sample_map = self.map_generator()
        
        game.set_doom_scenario_path(sample_map["wad_fname"])
        game.set_doom_map(sample_map["map_name"])
    
        game.init()

        sample = {"seed": sample_map["seed"], "map_name": sample_map["map_name"], "game": game}
        return sample

class DoomActionGenerator:
    def __init__(self, action_prob=0.3):
        super(DoomActionGenerator, self).__init__()
        self.action_prob = action_prob
    
    def __call__(self, num_available):
        return [random.uniform(0, 1) < self.action_prob for ii in range(num_available)]

class DoomStateGenerator:
    def __init__(self, game_generator, action_generator=None, advance=1, timeout=None):
        super(DoomStateGenerator, self).__init__()
        self.game_generator = game_generator
        self.action_generator = action_generator or DoomActionGenerator()
        self.advance = advance
        self.timeout = timeout
        
        self.game = None

    def __call__(self, new=False, action=None):
        sample = None
        while not sample:
            try:
                if not self.game or new or self.game.is_player_dead():
                    if self.game:
                        self.game.close()
                    game = self.game_generator()
                    self.seed, self.map_name, self.game = game["seed"], game["map_name"], game["game"]
                    new = True
                    action = None
                else:
                    if not action:
                        action = self.action_generator(self.game.get_available_buttons_size())
                    self.game.set_action(action)
                    self.game.advance_action(self.advance)
        
                state = self.game.get_state()
                if not state or (self.timeout and state.tic > self.timeout):
                    new = True
                    continue
        
                dead = self.game.is_player_dead()
                x = self.game.get_game_variable(vz.GameVariable.POSITION_X)
                y = self.game.get_game_variable(vz.GameVariable.POSITION_Y)
                z = self.game.get_game_variable(vz.GameVariable.POSITION_Z)
                angle = self.game.get_game_variable(vz.GameVariable.ANGLE)
                tic = state.tic / self.advance
            
                action = action or [0] * self.game.get_available_buttons_size()
                action = np.array([int(x) for x in action], dtype=np.uint8)
            
                sample = {
                    "seed": self.seed,
                    "map_name": self.map_name,
                    "new": new,
                    "dead": dead,
                    "xyz": np.array([x, y, z]),
                    "angle": angle,
                    "tic": tic,
                    "action": action,
                    "state": state
                }
            except vz.SignalException:
                if self.game:
                    try:
                        self.game.close()
                    except:
                        pass
                    self.game = None
        return sample

class BoringStateFilter():
    def __init__(self, generator, discard_first=3, boring_dist=64, boring_angle=30, max_boring_states=50, dead_is_boring=True):
        super(BoringStateFilter, self).__init__()
        self.generator = generator
        self.discard_first = discard_first
        self.boring_dist = boring_dist
        self.boring_angle = boring_angle
        self.max_boring_states = max_boring_states
        self.dead_is_boring = dead_is_boring

        self.states = []
        self.boring_states = 0
        
    def __call__(self, new=False, action=None):
        new2 = False
        actions = []
        discard_first = 0
        while True:
            new = new or self.boring_states >= self.max_boring_states
            sample = self.generator(new=new, action=action)
            x, y, z = sample["xyz"]
            angle = sample["angle"]
            if sample["new"]:
                new2, new = True, False
                actions = []
                self.states = []
                self.boring_states = 0
            elif hasattr(sample, "action"):
                actions.append(sample["action"])
            boring = sample["dead"] and self.dead_is_boring
            if not boring:
                for s in self.states:
                    dist_sq = ( - s[0]) ** 2 + (y - s[1]) ** 2 + (z - s[2]) ** 2
                    angle_diff = angle - s[3]
                    if dist_sq <= self.boring_dist ** 2 and ((angle_diff % 360 < self.boring_angle) or (-angle_diff % 360 < self.boring_angle)):
                        boring = True
                        break
            if not boring:
                self.boring_states = 0
                self.states.append((x, y, z, angle))
                if len(self.states) >= self.discard_first:
                    break
            else:
                self.boring_states += 1
        sample["new"] = new2
        del sample["action"]
        if actions:
            sample["actions"] = np.array(actions)
        return sample

class SemanticFilter:
    def __init__(self, generator, min_size=32, timeout=1000):
        super(SemanticFilter, self).__init__()

        self.generator = generator
        self.min_size = tuple(min_size) if isinstance(min_size, (tuple, list)) else (min_size, min_size)
        self.timeout = timeout

    def __call__(self, new=False, action=None):
        count = 0
        while True:
            sample = self.generator(new=new, action=action)
            state = sample["state"]
            for label in state.labels:
                if label.width > self.min_size[0] and label.height > self.min_size[1]:
                    return sample
            count += 1
            new = (count % self.timeout == 0)

class ImageProcessingFilter:
    def __init__(self, generator, scale=1, depth=True, types=True, instances=True):
        super(ImageProcessingFilter, self).__init__()

        self.generator = generator
        self.scale = int(scale)
        self.depth = depth
        self.types = types
        self.instances = instances

    def __call__(self, new=False, action=None):
        sample = self.generator(new=new, action=action)
        state = sample["state"]
        del sample["state"]
        screen = state.screen_buffer
        screen = screen.astype(np.float) / 255.0
        if self.scale != 1:
            screen = transform.resize(screen, (screen.shape[0] // self.scale, screen.shape[1] // self.scale), mode='reflect', anti_aliasing=True)
        screen = np.transpose(screen, (2, 0, 1))
        sample["color"] = screen
        if self.depth:
            depth = state.depth_buffer
            if self.scale != 1:
                # It would be nice to use transform.resize, but this doesn't seem to downsample properly :(
                depth = depth[((self.scale - 1) // 2)::self.scale, ((self.scale - 1) // 2)::self.scale]
            depth = depth.astype(np.float) * 2.0 / 255.0
            depth = np.expand_dims(depth, 0)
            sample["depth"] = depth
        if self.types:
            types = state.labels_buffer
            if self.scale != 1:
                types = types[((self.scale - 1) // 2)::self.scale, ((self.scale - 1) // 2)::self.scale]
            types = labels_to_types(state.labels, types)
            types = np.expand_dims(types, 0)
            sample["types"] = types
        if self.instances:
            instances = state.labels_buffer
            if self.scale != 1:
                instances = instances[((self.scale - 1) // 2)::self.scale, ((self.scale - 1) // 2)::self.scale]
            instances = labels_to_instances(state.labels, instances)
            instances = np.expand_dims(instances, 0)
            sample["instances"] = instances
        return sample

class ThreadedGeneratorWrapper:
    def __init__(self, generators, num_threads=1):
        self.generators = [generators(id) for id in range(num_threads)] if not isinstance(generators, (list, tuple)) else generators
        self.results = []

    def __call__(self):
        while not self.results:
            threads = []
            for generator in self.generators[1:]:
                thread = Thread(None, target=lambda: self.results.append(generator()))
                thread.start()
                threads.append(thread)
            self.results.append(self.generators[0]())
            for thread in threads:
                thread.join()
        return self.results.pop(0)

class SingletonFilter():
    def __init__(self, generator):
        super(SingletonFilter, self).__init__()
        self.generator = generator
        self.generated = False

    def __call__(self):
        if not self.generated:
            self.generated = True
            return self.generator()
        else:
            raise StopIteration

class DifferenceWrapper:
    def __init__(self, generator, delta=1):
        self.generator = generator
        self.delta = (delta, delta) if not isinstance(delta, (tuple, list)) else tuple(delta)
        self.buffer = []
        self.next_delta = self.delta[0]

    def __call__(self, new=False, action=None):
        while len(self.buffer) <= self.delta[1]:
            sample = self.generator(new=new, action=action)
            new = False
            if sample["new"]:
                self.buffer = []
            self.buffer.append(sample)
        new_sample = self.buffer[self.next_delta]
        old_sample = self.buffer[0]
        seed, map_name, new = old_sample["seed"], old_sample["map_name"], old_sample["new"]
        old_xyz, old_angle, old_tic = old_sample["xyz"], old_sample["angle"], old_sample["tic"]
        dead, new_xyz, new_angle, new_tic = new_sample["dead"], new_sample["xyz"], new_sample["angle"], new_sample["tic"] 
        actions = [sample["action"] for sample in self.buffer[:self.next_delta]]
        sample = {
            "seed": seed,
            "map_name": map_name,
            "new": new,
            "dead": dead,
            "old_xyz": old_xyz,
            "new_xyz": new_xyz,
            "delta_xyz": new_xyz - old_xyz,
            "old_angle": old_angle,
            "new_angle": new_angle,
            "delta_angle": new_angle - old_angle,
            "old_tic": old_tic,
            "new_tic": new_tic,
            "delta_tic": new_tic - old_tic
        }
        if actions:
            sample["actions"] = np.array(actions)
        for key in ["color", "depth", "types", "instances"]:
            if key in old_sample:
                sample["old_" + key] = old_sample[key]
            if key in new_sample:
                sample["new_" + key] = new_sample[key]

        self.next_delta += 1
        if self.next_delta > self.delta[1]:
            self.next_delta = self.delta[0]
            self.buffer.pop(0)

        return sample

class DoomDataset(Dataset):
    def __init__(self, path, size, generator, float_type, long_type, fname_fmt='{seed}-{map_name}-{xyz[0]:.0f}-{xyz[1]:.0f}-{xyz[2]:.0f}-{angle:.0f}-{tic}.png'):
        super(DoomDataset, self).__init__()
        if not os.path.exists(path):
            os.mkdir(path)
        self.path = path
        self.size = size
        self.generator = generator
        self.float_type = float_type
        self.long_type = long_type
        self.fname_fmt = fname_fmt
        self.files = os.listdir(path)
        if len(self.files) < self.size:
            self.generate(self.size - len(self.files))
        elif len(self.files) > self.size:
            self.remove(len(self.files) - self.size)

    def generate(self, num_samples=100):
        try:
            while num_samples > 0:
                sample = self.generator()
                metadata = {}
                subimages_data = []
                subimages = []
                for key in sample:
                    value = sample[key]
                    # Unfortunately, have to special case "action", as this can be a 2D array, but it's not an image
                    if isinstance(value, np.ndarray) and len(value.shape) > 1 and key.find("action") == -1:
                        subimage = value
                        subimage_data = {"name": key, "shape": list(subimage.shape)}
                        subimage = np.expand_dims(subimage, 0) if len(subimage.shape) == 2 else subimage
                        subimage = np.repeat(subimage, 3, 0) if subimage.shape[0] == 1 else subimage
                        subimage = np.transpose(subimage, (1, 2, 0))
                        if subimage.dtype == np.float:
                            subimage_data["type"] = "float"
                            subimage = (subimage * 255)
                        elif subimage.dtype == int:
                            subimage_data["type"] = "int"
                        else:
                            subimage_data["type"] = "byte"
                        subimage = subimage.astype(np.uint8)
                        subimages_data.append(subimage_data)
                        subimages.append(subimage)
                    else:
                        if isinstance(value, np.ndarray):
                            if value.dtype in [bool, np.uint8]:
                                value = value.astype(int)
                            metadata[key] = value.tolist()
                        else:
                            metadata[key] = value
                image = np.concatenate(subimages, axis=1)
                png = Image.fromarray(image)
                pnginfo = PngInfo()
                if metadata:
                    pnginfo.add_text('metadata', json.dumps(metadata))
                if subimages_data:
                    pnginfo.add_text('subimages', json.dumps(subimages_data))
                fname = self.fname_fmt.format(**sample)
                png.save('{}/{}'.format(self.path, fname), 'png', pnginfo=pnginfo)
                num_samples -= 1
        except StopIteration:
            pass
        self.files = os.listdir(self.path)        

    def remove(self, num_samples=100):
        for file in random.sample(self.files, num_samples):
            os.remove('{}/{}'.format(self.path, file))
            self.files.remove(file)

    def refresh(self, num_samples=None, fraction=0.1):
        num_samples = num_samples or int(self.size * fraction)
        self.remove(num_samples)
        self.generate(num_samples)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        png = PngImageFile('{}/{}'.format(self.path, fname))
        image = np.asarray(png, dtype=np.uint8)
        sample = {}
        if "metadata" in png.info:
            sample = json.loads(png.info["metadata"])
            for k in sample:
                if isinstance(sample[k], list):
                    sample[k] = np.array(sample[k])
        sample["fname"] = fname
        subimages = []
        if "subimages" in png.info:
            subimages = json.loads(png.info["subimages"])
        x = 0
        for subimage_data in subimages:
            subimage_name, subimage_shape, subimage_type = subimage_data["name"], subimage_data["shape"], subimage_data["type"]
            h, w = subimage_shape[len(subimage_shape) - 2:]
            subimage = image[:h, x:x+w]
            subimage = np.moveaxis(subimage, 2, 0)
            if len(subimage_shape) == 2:
                subimage = subimage[1, :, :]
            elif subimage_shape[0] == 1:
                subimage = np.expand_dims(subimage[1, :, :], 0)
            if subimage_type == "float":
                subimage = self.float_type(subimage.astype(np.float) / 255)
            elif subimage_type == "int":
                subimage = self.long_type(subimage.astype(int))
            sample[subimage_name] = subimage
            x += w
        return sample


from torch.autograd import Variable

def augment_mirror(sample):
    if random.randint(0, 1) == 1:
        for key in sample:
            value = sample[key]
            if isinstance(value, Variable):
                sample[key] = value.flip(-1)

def add_depth_mask(sample):
    sample_ext = {}
    for key in sample:
        if key.find("depth") != -1:
            sample_ext[key + '_mask'] = (sample[key] < 254/255).float()
    sample.update(sample_ext)