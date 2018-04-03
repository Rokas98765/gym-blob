from gym.envs.registration import register

register(
    id='blob-v0',
    entry_point='gym_blob.envs:blobEnv',
)
register(
    id='blob-extrahard-v0',
    entry_point='gym_blob.envs:blobExtraHardEnv',
)
