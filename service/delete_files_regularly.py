'''
Delete files which were created num_days ago.
'''


def main(): 
    num_days = 7   #day
    interval = 12*3600 #s
    while True:
        del_files_list = del_cache_old_time_files(num_days, files_path='./cache')
        time.sleep(interval) 
         

if __name__ == '__main__':
    import time
    from FilesManagement.files_management import del_cache_old_time_files
    main()
