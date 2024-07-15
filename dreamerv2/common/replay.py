import collections
import datetime
import io
import pathlib
import uuid
import os
import random
import pickle

import numpy as np

class Replay:
    def __init__(
        self,
        directory,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_endings=False,
        reservoir_sampling=False,
        num_tasks=1,
        recent_past_sampl_thres=0,
    ):
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity # replay buffer size
        self._ongoing = ongoing
        self._minlen = minlen # min eps length
        self._maxlen = maxlen # max length of sampled eps
        self._prioritize_endings = prioritize_endings # prioritizes the end of an epsiode which is loaded
        self._reservoir_sampling = reservoir_sampling
        self.recent_past_sampl_thres = recent_past_sampl_thres # probability threshold to trigger uniform episode
        self._num_tasks = num_tasks  # the number of tasks in the cl loop
        self._random = np.random.RandomState()
        
        self._complete_eps, self._tasks, self._reward_eps = self.load_episodes(directory=self._directory, capacity=capacity, minlen=minlen)
        self._ongoing_eps = collections.defaultdict(lambda: collections.defaultdict(list)) # current episodes not yet finished and saved
        self._total_episodes, self._total_steps = self.count_episodes(directory) # total eps/steps seen during all the training
        self._loaded_episodes = len(self._complete_eps) # eps/steps currently loaded in the buffer
        self._loaded_steps = sum(ep_length(x) for x in self._complete_eps.values())

        self._plan2explore = None
        self.set_task()
                
    def set_task(self, task_idx=0):
        self.task_idx = task_idx

    def get_task(self):
        return self.task_idx
                 
    @property
    def stats(self):
        ret = {
            'total_steps': self._total_steps,
            'total_episodes': self._total_episodes,
            'loaded_steps': self._loaded_steps,
            'loaded_episodes': self._loaded_episodes,
            'avg_task': np.mean([v for k, v in self._tasks.items()]) if self._tasks else None,
        }
        return ret
    
    def add_step(self, transition, worker=0):
        """
        Add a step to the current episode (not saved yet)
        """
        episode = self._ongoing_eps[worker]
        for key, value in transition.items():
            episode[key].append(value)
        if transition['is_last']:
            self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode):
        """
        When a terminal step is loaded, save the full episode in the buffer
        """
        length = ep_length(episode)
        
        # Skip short episodes
        if length < self._minlen:
            print(f'Skipping short episode of length {length}.')
            return
        
        # Update counters
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        
        # Save the episode locally
        episode = {key: convert(value) for key, value in episode.items()}
        task = self.get_task()
        filename = self.save_episode(self._directory, episode, task, self._total_episodes)
        
        # Add the episode to the buffer
        self._complete_eps[str(filename)] = episode
        self._tasks[str(filename)] = task
        self._reward_eps[str(filename)] = episode['reward'].astype(np.float64).sum()

        # Reservoir sampling
        if self._reservoir_sampling:
            if self._loaded_steps < self._capacity:
                self._complete_eps[str(filename)] = episode
            else:
                # Sample a random episode from the buffer
                idx = np.random.randint(self._total_episodes)
                
                # Remove it with probability current_size/max_size
                if idx < self._loaded_episodes:
                    filenames = [k for k, v in self._complete_eps.items()]
                    filename_to_remove = filenames[idx]
                    episode_to_remove = self._complete_eps[str(filename_to_remove)]
                    self.remove_episode(filename_to_remove, episode_to_remove)
            
        if self._capacity:
            # Remove random episodes if the buffer is full
            while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
                if self._reservoir_sampling: # remove a random one with rs
                    candidate, episode = random.sample(self._complete_eps.items(), 1)[0]
                else: # otherwise the oldest
                    candidate, episode = next(iter(self._complete_eps.items()))  
                self.remove_episode(candidate, episode)
        
        # Save the rs buffer locally
        if self._reservoir_sampling:
            with open(self._directory/f'rs_buffer.pkl', 'wb') as file:
                pickle.dump(list(self._complete_eps.keys()), file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def remove_episode(self, filename_to_remove, episode_to_remove):
        """
        Remove an episode from the buffer
        """
        # Update counters
        self._loaded_steps -= ep_length(episode_to_remove)
        self._loaded_episodes -= 1
        
        # Remove the episode from the buffers
        del self._complete_eps[str(filename_to_remove)]
        del self._tasks[str(filename_to_remove)]
        del self._reward_eps[str(filename_to_remove)]
    
    def dataset(self, batch, length):
        """
        Create a generator from the saved episodes to use as dataset for the model
        """
        generator = iter(self._generate_chunks(length))
        dataset = self.from_generator(generator, batch)
        return dataset
    
    def from_generator(self, generator, batch_size):
        """
        Yield batches from the input generator
        """
        while True:
            batch = []
            for _ in range(batch_size):
                batch.append(next(generator))
            data = {}
            for key in batch[0].keys():
                data[key] = []
                for i in range(batch_size):
                    data[key].append(batch[i][key])
                data[key] = np.stack(data[key], 0)
            yield data

    def _generate_chunks(self, length):
        """
        Generate fixed size sequences from the buffer
        """
        sequence = self._sample_sequence()
        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding['action'])
                if len(sequence['action']) < 1:
                    sequence = self._sample_sequence()
            chunk = {k: np.concatenate(v) for k, v in chunk.items()}
            yield chunk

    def _sample_sequence(self):
        """
        Sample a sequence from an episode (not necessarly a full episode)
        """
        episodes_keys = list(self._complete_eps.keys())
        if self._ongoing:
            episodes_keys += [k for k, v in self._ongoing_eps.items() if ep_length(v) >= self._minlen]
        
        # 50:50 sampling if recent_past_sampl_thres is 0.5
        if self.recent_past_sampl_thres > np.random.random():
            # Sample using a triangular distribution favouring latest experience
            episode_key = episodes_keys[int(np.floor(np.random.triangular(0, len(episodes_keys), len(episodes_keys), 1)))]
        else:
            episode_key = self._random.choice(episodes_keys)
        episode = self._complete_eps[episode_key]
        info = self.parse_episode_name(episode_key)
        self._logger.scalar("replay/total_episode", info['total_episodes'])
        self._logger.scalar("replay/task", info['task'])

        total = len(episode['action'])
        length = total
        if self._maxlen:
            length = min(length, self._maxlen)
            
        # Randomize length to avoid all chunks ending at the same
        length -= np.random.randint(self._minlen)
        length = max(self._minlen, length)
        upper = total - length + 1 # last transition of the episode
        
        if self._prioritize_endings:
            upper += self._minlen
        index = min(self._random.randint(upper), total - length)
        sequence = {
            k: convert(v[index: index + length])
            for k, v in episode.items() if not k.startswith('log_')}
        sequence['is_first'] = np.zeros(len(sequence['action']), bool)
        sequence['is_first'][0] = True
        if self._maxlen:
            assert self._minlen <= len(sequence['action']) <= self._maxlen
        return sequence

    def count_episodes(self, directory):
        filenames = list(directory.glob("*.npz"))
        num_episodes = len(filenames)
        num_steps = sum(int(str(os.path.basename(n)).split("-")[3]) - 1 for n in filenames)
        return num_episodes, num_steps

    def save_episode(self, directory, episode, task, total_episodes):
        """
        Save an episode locally with info
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        identifier = str(uuid.uuid4().hex)
        length = ep_length(episode)
        filename = directory / f'{timestamp}-{identifier}-{task}-{length}-{total_episodes}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())
        return filename

    def load_episodes(self, directory, capacity=None, minlen=1, coverage_sampling=False, check=False):
        """
        Load locally saved episodes in the buffer
        """

        # Retrieve all saved episodes
        filenames = sorted(directory.glob('*.npz'))

        if capacity:
            num_steps = 0
            num_episodes = 0
            # If using reservoir sampling, a random shuffle of the replay buffer will preserve and uniform distribution over all tasks
            if os.path.exists(directory/f'rs_buffer.pkl'):
                with open(directory/f'rs_buffer.pkl', 'rb') as handle:
                    filenames = pickle.load(handle)
                filenames = [pathlib.Path(filename) for filename in filenames]
            # Take only most recent episodes
            for filename in reversed(filenames):
                length = int(str(os.path.basename(filename)).split('-')[3])
                num_steps += length
                num_episodes += 1
                if num_steps >= capacity:
                    break
            filenames = filenames[-num_episodes:]
            
        episodes = {}
        tasks = {}
        rewards_eps = {}
        for filename in filenames:
            try:
                with filename.open('rb') as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f'Could not load episode {str(filename)}: {e}')
                continue
            episodes[str(filename)] = episode
            task = int(str(os.path.basename(filename)).split('-')[2])
            tasks[str(filename)] = task
            rewards_eps[str(filename)] = episode['reward'].astype(np.float64).sum()

        return episodes, tasks, rewards_eps

    def parse_episode_name(self, episode_name):
        episode_name = os.path.basename(episode_name)
        parts = episode_name.split("-")
        timestamp = parts[0]
        identifier = parts[1]
        task = parts[2]
        length = parts[3]
        total_episodes = parts[4] if len(parts) == 5 else None
        if len(parts) == 5:
            total_episodes = total_episodes.split(".")[0]
        elif len(parts) == 4:
            length = length.split(".")[0]

        return {
            "timestamp": timestamp,
            "identifier": identifier,
            "task": int(task),
            "length": int(length),
            "total_episodes": int(total_episodes) if total_episodes else np.nan,
        }
    
    
    @property
    def agent(self):
        return self._plan2explore

    @agent.setter
    def agent(self, plan2explore):
        self._plan2explore = plan2explore

    @property
    def logger(self):
        return self._logger

    @agent.setter
    def logger(self, logger):
        self._logger = logger 
    
def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value

def ep_length(episode):
    return len(episode['action']) - 1
