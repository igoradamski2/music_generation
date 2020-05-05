from songs_utils import *

def select_random_song(linkObj):
	
	random_song = random.sample(linkObj.links, 1)
	song_index  = linkObj.links.index(random_song[0])
	if (linkObj.names[song_index]) in linkObj.used_names.keys():
		name = (linkObj.names[song_index]) + str(linkObj.used_names[linkObj.names[song_index]])
		linkObj.used_names[linkObj.names[song_index]] += 1
	else:
		name = (linkObj.names[song_index])
		linkObj.used_names[(linkObj.names[song_index])] = 1

	name = name.replace("/", "-")
	
	return linkObj, random_song[0], name



if __name__ == "__main__":

	file = 'maestro-v2.0.0/maestro-v2.0.0.csv'
	what_type = 'train'
	link = '2017/MIDI-Unprocessed_050_PIANO050_MID--AUDIO-split_07-06-17_Piano-e_3-01_wav--4.midi'
	name = 'test4'
	contextLength = 100 # in timesteps
	length = 60
	numRegions = 4
	regionLengths = [2, 5, 7, 3]

	params = Config(Config.__read_config_file__('generation_config.json'))
	params = params.generation
	params.batch_size = 1
	params.articulation_prob = 0.01

	linkObj = DataLinks(file, what_type, 10, 10)
	linkObj.used_names = {}

	for i in range(2):
		linkObj, link, name = select_random_song(linkObj)
		for art_prob in [0.5, 0.4, 0.3, 0.15, 0.05]:
			for raw_mult in [0.8, 1, 1.5, 2]:	
				for modify_raw in [True, False]:
					name2 = name + '_artic=' +str(art_prob) + '_raw_mult='+str(raw_mult)+'_modify='+str(int(modify_raw))
					
					params.articulation_prob = art_prob
					params.raw_multiplier = raw_mult
					params.modify_raw = modify_raw

					curr_song = FakeSong(file = file, 
										 what_type = what_type,
										 link = link, 
										 name = name2,
										 numRegions = numRegions,
										 contextLength = contextLength,
										 length = length, 
										 regionLengths = regionLengths)
					
					params.how = 'raw'
					curr_song.fill_gaps(params)
					curr_song.save_songs(name2)
		

				
				



	
	
