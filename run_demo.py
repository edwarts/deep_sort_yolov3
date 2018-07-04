import demo4 as demo

def timestamp_convert(filename):
    date = filename.split("-")[1]
    time = filename.split("-")[2]
    dt = date[0:4] + '-' + date[4:6] + '-' + date[6:8] + ' ' + time[0:2] + ':' + time[2:] + ':00'
    ts = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp()
    return ts

def process_video_files_from_folder(root_folder,output_folder):
    # full_categories = [x[0] for x in os.walk(root_folder) if x[0]][1:]
    print(root_folder)
    data = []
    filenames=next(os.walk(root_folder))[2]

    #print(filenames)
    for file_name in filenames:
        first_part=file_name.split(".")[0]
        cap_name = file_name
        jfile = first_part + '.json'
        cam_id = first_part.split("-")[0]
        start_time = timestamp_convert(first_part)
        #print({'cap_name': cap_name, 'jfile_name': jfile, 'cam_id': cam_id,
        #         'timestamp': start_time})
        # video_to_json(root_folder +'/'+cap_name, output_folder +'/'+jfile, cam_id, start_time)
        data.append(
                {'cap_name': cap_name, 'jfile_name': jfile, 'cam_id': cam_id,
                 'timestamp': start_time})

    return data


if __name__ == "__main__":
    samples_per_sec=5
    
    if len(sys.argv)==3:
    #     process whole folder
        args_all = process_video_files_from_folder(sys.argv[1],sys.argv[2])
        print(args_all)
        root_folder = sys.argv[1]
        output_folder = sys.argv[2]
        # boringtao: process all videos in the folder
        for args in args_all:
            cap_name =  root_folder+'/'+args['cap_name']
            jfile = output_folder+'/'+args['jfile_name']
            cam_id = args['cam_id']
            start_time = args['timestamp']
            demo.video_to_json(cap_name, jfile, cam_id, start_time, samples_per_sec)
    else :
        parser = argparse.ArgumentParser(description='Video to json')
        parser.add_argument('cap_name', help='Path to input video file.')
        parser.add_argument('jfile', help='Path to JSON output file.')
        parser.add_argument('cam_id', help='Camera id.')
        parser.add_argument('start_time', help='Start timestamp.')
        args=parser.parse_args()
        root_folder='datasets/videos/test'
        output_folder='datasets/videos/test'
        cap_name = root_folder+'/'+args.cap_name
        jfile = output_folder+'/'+args.jfile
        cam_id = args.cam_id
        start_time = args.start_time
        print(cap_name)
        print(jfile)
        print(cam_id)
        print(start_time)
        demo.video_to_json(cap_name, jfile, cam_id, start_time, samples_per_sec)
    
    