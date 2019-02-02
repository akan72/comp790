'''
Created on Jun 12, 2012

@author: yyb
'''


'Write the information into GraphML format files'


import os;
import ReadData;
from copy import deepcopy;


ReadData.readLinks();

myHourDict = ReadData.hourDict;
myDayDict = ReadData.dayDict;
myMonDict = ReadData.monDict;
myYearDict = ReadData.yearDict;
myLinkDict = ReadData.linkDict;

totalHour = len(ReadData.hourList);
totalDay = len(ReadData.dayList);
totalMon = len(ReadData.monList);
totalYear = len(ReadData.yearList);


#===============================================================================
# Write information into GraphML format files with an hourly unit
#===============================================================================
def writeHourGML(size):
    fPath = '../GraphML/span(' + str(size) + 'hour(s))/';
    if os.path.exists(fPath) != 1:
        os.mkdir(fPath);
    i = 1;
    myHourList = ReadData.hourList[:];
    myNdDict = deepcopy(ReadData.nodeDict);
    while len(myHourList) != 0:
        wd = myHourList[0:size];
        del myHourList[0:size];
        fname = fPath + str(i) + '.graphml';
        fobj = open(fname, 'w');
        fobj.write('<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n\txsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n\t  http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n\n');
        fobj.write('   <key id="d1" for="edge" attr.name="timestamp" attr.type="string"/>\n');
        fobj.write('   <key id="d_n" for="node" attr.name="modification" attr.type="string">\n' + '     <default>add</default>\n' + '   </key>\n');
        fobj.write('   <key id="d_e" for="edge" attr.name="modification" attr.type="string">\n' + '     <default>add</default>\n' + '   </key>\n\n');
        fobj.write('  <graph id="' + str(i) + '" edgedefault="directed">\n');
        
        for hour in wd:
            li = myHourDict[hour];
            for (fromNd, toNd) in li:
                # write node
                if myNdDict[fromNd] == 0:
                    myNdDict[fromNd] = 1;
                    fobj.write('\t<node id="' + str(fromNd) + '"/>\n');
                if myNdDict[toNd] == 0:
                    myNdDict[toNd] = 1;
                    fobj.write('\t<node id="' + str(toNd) + '"/>\n');
                # write edge
                fobj.write('\t<edge source="' + str(fromNd) + '" target="' + str(toNd) + '">\n');
                fobj.write('\t\t<data key="d1">' + myLinkDict[(fromNd, toNd)] + '</data>\n');
                fobj.write('\t</edge>\n')
        
        fobj.write('  </graph>\n');
        fobj.write('</graphml>');
        fobj.close();
        i += 1;


#===============================================================================
# Write information into GraphML format files with an daily unit
#===============================================================================
def writeDayGML(size):
    fPath = '../GraphML/span(' + str(size) + 'day(s))/';
    if os.path.exists(fPath) != 1:
        os.mkdir(fPath);
    i = 1;
    myDayList = ReadData.dayList[:];
    myNdDict = deepcopy(ReadData.nodeDict);
    while len(myDayList) != 0:
        wd = myDayList[0:size];
        del myDayList[0:size];
        fname = fPath + str(i) + '.graphml';
        fobj = open(fname, 'w');
        fobj.write('<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n\txsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n\t  http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n\n');
        fobj.write('   <key id="d1" for="edge" attr.name="timestamp" attr.type="string"/>\n');
        fobj.write('   <key id="d_n" for="node" attr.name="modification" attr.type="string">\n' + '     <default>add</default>\n' + '   </key>\n');
        fobj.write('   <key id="d_e" for="edge" attr.name="modification" attr.type="string">\n' + '     <default>add</default>\n' + '   </key>\n\n');
        fobj.write('  <graph id="' + str(i) + '" edgedefault="directed">\n');
        
        for day in wd:
            li = myDayDict[day];
            for (fromNd, toNd) in li:
                # write node
                if myNdDict[fromNd] == 0:
                    myNdDict[fromNd] = 1;
                    fobj.write('\t<node id="' + str(fromNd) + '"/>\n');
                if myNdDict[toNd] == 0:
                    myNdDict[toNd] = 1;
                    fobj.write('\t<node id="' + str(toNd) + '"/>\n');
                # write edge
                fobj.write('\t<edge source="' + str(fromNd) + '" target="' + str(toNd) + '">\n');
                fobj.write('\t\t<data key="d1">' + myLinkDict[(fromNd, toNd)] + '</data>\n');
                fobj.write('\t</edge>\n')
        
        fobj.write('  </graph>\n');
        fobj.write('</graphml>');
        fobj.close();
        i += 1;


#===============================================================================
# Write information into GraphML format files with an monthly unit
#===============================================================================
def writeMonthGML(size):
    fPath = '../GraphML/span(' + str(size) + 'month(s))/';
    if os.path.exists(fPath) != 1:
        os.mkdir(fPath);
    i = 1;
    myMonList = ReadData.monList[:];
    myNdDict = deepcopy(ReadData.nodeDict);
    while len(myMonList) != 0:
        wd = myMonList[0:size];
        del myMonList[0:size];
        fname = fPath + str(i) + '.graphml';
        fobj = open(fname, 'w');
        fobj.write('<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n\txsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n\t  http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n\n');
        fobj.write('   <key id="d1" for="edge" attr.name="timestamp" attr.type="string"/>\n');
        fobj.write('   <key id="d_n" for="node" attr.name="modification" attr.type="string">\n' + '     <default>add</default>\n' + '   </key>\n');
        fobj.write('   <key id="d_e" for="edge" attr.name="modification" attr.type="string">\n' + '     <default>add</default>\n' + '   </key>\n\n');
        fobj.write('  <graph id="' + str(i) + '" edgedefault="directed">\n');
        
        for mon in wd:
            li = myMonDict[mon];
            for (fromNd, toNd) in li:
                # write node
                if myNdDict[fromNd] == 0:
                    myNdDict[fromNd] = 1;
                    fobj.write('\t<node id="' + str(fromNd) + '"/>\n');
                if myNdDict[toNd] == 0:
                    myNdDict[toNd] = 1;
                    fobj.write('\t<node id="' + str(toNd) + '"/>\n');
                # write edge
                fobj.write('\t<edge source="' + str(fromNd) + '" target="' + str(toNd) + '">\n');
                fobj.write('\t\t<data key="d1">' + myLinkDict[(fromNd, toNd)] + '</data>\n');
                fobj.write('\t</edge>\n')
        
        fobj.write('  </graph>\n');
        fobj.write('</graphml>');
        fobj.close();
        i += 1;


#===============================================================================
# Write information into GraphML format files with an yearly unit
#===============================================================================
def writeYearGML(size):
    fPath = '../GraphML/span(' + str(size) + 'year(s))/';
    if os.path.exists(fPath) != 1:
        os.mkdir(fPath);
    i = 1;
    myYearList = ReadData.yearList[:];
    myNdDict = deepcopy(ReadData.nodeDict);
    while len(myYearList) != 0:
        wd = myYearList[0:size];
        del myYearList[0:size];
        fname = fPath + str(i) + '.graphml';
        fobj = open(fname, 'w');
        fobj.write('<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n\txsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n\t  http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n\n');
        fobj.write('   <key id="d1" for="edge" attr.name="timestamp" attr.type="string"/>\n');
        fobj.write('   <key id="d_n" for="node" attr.name="modification" attr.type="string">\n' + '     <default>add</default>\n' + '   </key>\n');
        fobj.write('   <key id="d_e" for="edge" attr.name="modification" attr.type="string">\n' + '     <default>add</default>\n' + '   </key>\n\n');
        fobj.write('  <graph id="' + str(i) + '" edgedefault="directed">\n');
        
        for yr in wd:
            li = myYearDict[yr];
            for (fromNd, toNd) in li:
                # write node
                if myNdDict[fromNd] == 0:
                    myNdDict[fromNd] = 1;
                    fobj.write('\t<node id="' + str(fromNd) + '"/>\n');
                if myNdDict[toNd] == 0:
                    myNdDict[toNd] = 1;
                    fobj.write('\t<node id="' + str(toNd) + '"/>\n');
                # write edge
                fobj.write('\t<edge source="' + str(fromNd) + '" target="' + str(toNd) + '">\n');
                fobj.write('\t\t<data key="d1">' + myLinkDict[(fromNd, toNd)] + '</data>\n');
                fobj.write('\t</edge>\n')
        
        fobj.write('  </graph>\n');
        fobj.write('</graphml>');
        fobj.close();
        i += 1;
        

#===============================================================================
# The main function
#===============================================================================
def main():
    while True:
        print 'Choose a basic unit to stream the Facebook-Growth dataset (enter 1, 2, 3, 4 or 5):\n1.hourly\n2.daily\n3.monthly\n4.yearly\n5.Quit\n',
        choice = int(raw_input());
        if choice == 1:
            print 'There are ' + str(totalHour) + ' hourly instances in this dataset.';  
            print 'Enter an integer as a time window size (1 ~', totalHour, '):';
            windowSize = int(raw_input());
            writeHourGML(windowSize);
        elif choice == 2:
            print 'There are ' + str(totalDay) + ' dailyly instances in this dataset';  
            print 'Enter an integer as a time window size (1 ~', totalDay, '):';
            windowSize = int(raw_input());
            writeDayGML(windowSize);
        elif choice == 3:
            print 'There are ' + str(totalMon) + ' monthly instances in this dataset';  
            print 'Enter an integer as a time window size (1 ~', totalMon, '):';
            windowSize = int(raw_input());
            writeMonthGML(windowSize);
        elif choice == 4:
            print 'There are ' + str(totalYear) + ' yearly instances in this dataset';  
            print 'Enter an integer as a time window size (1 ~', totalYear, '):';
            windowSize = int(raw_input());
            writeYearGML(windowSize);
        elif choice == 5:
            break;
        else:
            print 'Bad choice';
            continue;
            
        opt = raw_input('\nTry another time window size (yes or no)?\n');
        if opt.strip()[0].lower() == 'n':
            print 'Terminating......';
            break;
        elif opt.strip()[0].lower() == 'y':
            continue;
        else:
            print 'Invalid input, terminating......';
            break;


#===============================================================================
# Execution
#===============================================================================
if __name__ == '__main__':
    main();
    
    
    
    