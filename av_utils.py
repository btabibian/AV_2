write=None
def av_debug(msg):
    print msg
def write_array_to_file(data):
    global write
    if(write==None):
        write=file('output.txt','a');
    for i in data:
        write.write(str(i)+',')
    write.write('\r\n')
    write.flush()
    
