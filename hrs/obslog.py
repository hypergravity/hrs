#!/usr/bin/env python
import os
import datetime
import dateutil
import pylab

OBJECT_LST = ['Bias','Flat','Comp','Dark']
CALIB_LST  = ['ThAr','Iodn','Mo','FeAr']

class LogItem(object):
    def __init__(self,**kwargs):
        for name in kwargs:
            value = kwargs[name]
            object.__setattr__(self,name,value)

    def get_filename(self,filename_composition):
        '''get filename for given item id
        %Y Year with century as a decimal number.
        %m Month as a decimal number [01,12].
        %d Day of the month as a decimal number [01,31].
        %iN Frame ID as a decimal number, with length of N filled by zero.
        '''
        fn_lst = []
        i=0
        while(True):
            if i==len(filename_composition):
                break
            elif filename_composition[i]=='%':
                key = filename_composition[i+1]
                if key == 'Y':
                    fn_lst.append(str(self.date.year))
                    i += 2
                elif key == 'm':
                    fn_lst.append(str(self.date.month).rjust(2,'0'))
                    i += 2
                elif key == 'd':
                    fn_lst.append(str(self.date.day).rjust(2,'0'))
                    i += 2
                elif key == 'i':
                    id_len = int(filename_composition[i+2])
                    fn_lst.append(str(self.id).rjust(id_len,'0'))
                    i += 3
            else:
                fn_lst.append(filename_composition[i])
                i += 1
        return ''.join(fn_lst)

    def get_datetime(self,date):
        '''get actually datetime'''
        if self.time != '':
            g = self.time.split(':')
            if int(g[0])>23:
                g[0] = str(int(g[0])-24).rjust(2,'0')
                tstring = ':'.join(g)
                t = dateutil.parser.parse(tstring).time()
                logt = datetime.datetime.combine(date+datetime.timedelta(days=1),t)
            else:
                t = dateutil.parser.parse(self.time).time()
                logt = datetime.datetime.combine(date,t)
            return logt
        else:
            return None
 

class Log(object):
    def __init__(self,**kwargs):
        self.item_list = kwargs.pop('list',[])

    def add_item(self,item):
        self.item_list.append(item)

    def add_itemlist(self,list):
        self.item_list = list

    def check_files(self):
        #check file exists
        for item in self.item_list:
           if not item.skip:
                filename = item.get_filename(self.filename_composition)
                if not os.path.exists(filename):
                    print 'Error', filename, 'does not exist'
                    return False
        return True

    def show_items(self):
        for item in self.item_list:
            if not item.skip:
                print item.id, item.date, item.object,
                print item.exptime, item.note, item.i2,item.skip

    def __str__(self):
        string_lst = []
        for item in self.item_list:
            lst = [str(item.id),
                   item.object.rjust(6),
                   item.date,
                   str(item.exptime).rjust(8),
                  ]
            string_lst.append(' '.join(lst))
        return os.linesep.join(string_lst)

    def save_file(self,filename,object=None,exptime=None):
        file = open(filename,'w')
        for item in self.item_list:
            if not item.skip:
                if object == 'Star':
                    if ((item.object in OBJECT_LST) or
                        (item.object in  CALIB_LST)):
                        continue
                elif object != None and item.object!=object:
                    continue
                if exptime != None and abs(item.exptime-exptime)>1e-6:
                    continue
                filename = item.get_filename(self.filename_composition)
                file.write(filename+os.linesep)
        file.close()

    def get_filenamelist(self,**kwargs):
        lst = []
        for item in self.item_list:
            match = True
            for key in kwargs:
                value = kwargs[key]

                # judge string
                if type(value) == type('a'):
                    this_match = getattr(item,key).lower() == value.lower()

                # judge integer or long integer
                elif type(value) == type(1) or type(value) == type(1L):

                    # exception: if type do not match each other
                    # item attr is float but compare value is integer
                    if type(getattr(item,key)) == type(1.0):
                        # convert value to float
                        value = float(value)
                        this_match = abs(getattr(item,key)-value) < 1e-6
                    else:
                        this_match = getattr(item,key) == value

                # judge float
                elif type(value) == type(1.0):
                    this_match = abs(getattr(item,key)-value) < 1e-6

                else:
                    this_match = False

                match = match and this_match

            if match:
                lst.append(item.get_filename(self.filename_composition))

        return lst

    def set_filename_composition(self,list):
        self.filename_composition=list

    def check_file_exist(self):
        label = True
        for item in self.item_list:
            if not item.skip:
                filename = item.get_filename(self.filename_composition)
                if not os.path.exists(filename):
                    print 'Error:', filename, 'doe not exist'
                    label = False
        return label

def read_log(filename):

    '''read log file, return a Log instance '''

    object_dict = {
    'bias':'Bias',
    'flat':'Flat',
    'dark':'Dark',
    'i2':  'Iodn',
    'iodn':'Iodn',
    'comp':'Comp',
    'thar':'ThAr',
    }

    logfile = open(filename)

    log = Log()

    for line in logfile:
        line = line.strip()

        # blank line
        if len(line)==0:
            continue

        # read header
        if line[0]=='%':
            line = line[1:].strip()
            g = line.split('=')

            # read columns information
            if g[0].strip().lower()=='cols':
                col_lst = []
                tmp_g = g[1].split(',')
                for e in tmp_g:
                    col_lst.append(e.strip().lower())

            # read obs date
            elif g[0].strip().lower()=='date':
                date = g[1].strip()
                log.date = dateutil.parser.parse(date)

            # read filename composition
            elif g[0].strip().lower()=='filename composition':
                log.filename_composition = g[1].strip()

            # read program name
            elif g[0].strip().lower()=='program':
                log.program = g[1].strip()

            # read observatory
            elif g[0].strip().lower()=='observatory':
                log.observatory = g[1].strip()

            # read telescope
            elif g[0].strip().lower()=='telescope':
                log.telescope = g[1].strip()

            # read instrument
            elif g[0].strip().lower()=='instrument':
                log.instrument = g[1].strip()

            # read detector
            elif g[0].strip().lower()=='detector':
                log.detector = g[1].strip()

            # read observer
            elif g[0].strip().lower()=='observer':
                log.observer = g[1].strip()

            # read operator
            elif g[0].strip().lower()=='operator':
                log.operator = g[1].strip()

            # read timezone
            elif g[0].strip().lower()=='time zone':
                log.timezone = g[1].strip()

        #elif line[0]!='#':
        else:
            g = line.split('|')
            if g[0].strip()=='#':
                skip = True
            else:
                skip = False

            pos = col_lst.index('id')
            idg = g[pos+1].split(',')
            id_lst = []
            for idge in idg:
                ids = idge.split('-')
                if len(ids)==2:
                    for i in range(int(ids[0]),int(ids[1])+1):
                        id_lst.append(i)
                else:
                    id_lst.append(int(idge))

            # generate tmp item list
            item_lst = []
            for id in id_lst:
                item = LogItem(id=id,date=log.date,skip=skip)
                item_lst.append(item)

            # find object
            pos = col_lst.index('object')
            obj_str = g[pos+1]
            if len(id_lst)>1:
                obj_g = obj_str.split('x')
                object = obj_g[0].strip()
                count = int(obj_g[1])
                if count != len(id_lst):
                    print "Error: count and id don't match",obj_str,id_lst
            else:
                object = obj_str.strip()

            if object.strip().lower() in object_dict:
                object = object_dict[object.strip().lower()]

            for item in item_lst:
                item.object = object

            # find time
            if 'time' in col_lst:
                pos = col_lst.index('time')
                time = g[pos+1].strip()
                for item in item_lst:
                    item.time = time
                    # item.time is a string

            # find exptime 
            pos = col_lst.index('exptime')
            if object == 'Bias':
                exptime = 0.0
            elif g[pos+1].strip()=='':
                exptime = None
            else:
                exptime = float(g[pos+1])
            for item in item_lst:
                item.exptime = exptime

            # find note
            if 'note' in col_lst:
                pos = col_lst.index('note')
                note = g[pos+1].strip()
                if 'with i2' in note.lower():
                    i2 = True
                    p = note.lower().index('with i2')
                    l = len('with i2')
                    note = note[:p]+note[p+l:]
                elif 'without i2' in note.lower():
                    i2 = False
                    p = note.lower().index('without i2')
                    l = len('without i2')
                    note = note[:p]+note[p+l:]
                else:
                    i2 = None
                note = note.strip()
                if len(note)==0:
                    note = None

            for item in item_lst:
                item.note = note
                item.i2 = i2

            for item in item_lst:
                log.add_item(item)
                #print item.date, item.id, item.object, item.time, item.exptime, item.note, item.i2
    logfile.close()

    return log

def find_logfile():

    '''find log file with surfix '.obslog' in current directory'''

    log_lst = []
    surfix  = '.obslog'

    for filename in os.listdir(os.curdir):
        if filename[-len(surfix):] == surfix:
            log_lst.append(filename)

    if len(log_lst)==1:
        return log_lst[0]
    else:
        print log_lst
        return None

