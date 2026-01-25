import os

def get_video_directories(path):
    """
    Get list of video directories. 
    Returns subdirectories in data/videos/, or ["."] if videos are directly in data/videos/
    """
    data_path = os.path.join(path, "videos")
    if not os.path.exists(data_path):
        return []
    
    try:
        all_items = os.listdir(data_path)
    except PermissionError:
        return []
    
    # Check if there are subdirectories (exclude hidden files/dirs)
    subdirs = [d for d in all_items 
               if os.path.isdir(os.path.join(data_path, d)) 
               and not d.startswith('.')]
    
    # Check if there are video files directly in data/videos/
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    direct_files = [f for f in all_items 
                   if os.path.isfile(os.path.join(data_path, f)) 
                   and os.path.splitext(f)[1].lower() in video_extensions]
    
    # Sort subdirectories for consistent display
    subdirs = sorted(subdirs)
    
    # If there are direct video files, add "." as an option (with better label)
    if direct_files:
        if subdirs:
            # Return with "." first, then subdirectories
            return [".", *subdirs]  # "." means current directory (data/videos/)
        else:
            return ["."]  # Only direct files, no subdirectories
    else:
        return subdirs  # Only subdirectories, no direct files

def get_video_files(path, directory):
    """
    Get video files from a directory.
    If directory is ".", look directly in data/videos/
    Otherwise, look in data/videos/directory/
    """
    if not directory:
        return []
    
    data_path = os.path.join(path, "videos")
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    if directory == ".":
        # Look directly in data/videos/
        search_path = data_path
    else:
        # Look in data/videos/directory/
        search_path = os.path.join(data_path, directory)
    
    if not os.path.exists(search_path):
        return []
    
    files = [f for f in os.listdir(search_path) 
            if os.path.isfile(os.path.join(search_path, f)) 
            and os.path.splitext(f)[1].lower() in video_extensions]
    
    return sorted(files)