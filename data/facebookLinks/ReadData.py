'''
Created on Jun 12, 2012

@author: yyb
'''

'Read information from the original data file'


from datetime import datetime;


hourDict = {};      # a dictionary for storing hours and the links created in those hours
dayDict = {};       # a dictionary for storing days and the links created in those days
monDict = {};       # a dictionary for storing months and the links created in those months
yearDict = {};      # a dictionary for storing years and the links created in those years

nodeDict = {};      # a dictionary for storing nodes
linkDict = {};      # a dictionary for storing edges (links)

hourList = [];
dayList = [];
monList = [];
yearList = [];




#===============================================================================
# Read links and their creating time from the original file
#===============================================================================
def readLinks():
    fobj = open('../Raw Data/facebook-links.txt.anon', 'r');
    while True:
        s = fobj.readline();
        if len(s) == 0:
            break;
        [fromNode, toNode, tmstmp] = s.split('\t');     # split a line into 3 parts
        if tmstmp != '\\N\n':
            fromNode = int(fromNode); toNode = int(toNode); tmstmp = int(tmstmp);
            # insert a node into the dictionary
            if fromNode not in nodeDict:
                nodeDict[fromNode] = 0;
            if toNode not in nodeDict:
                nodeDict[toNode] = 0;
            tmstmp = datetime.utcfromtimestamp(tmstmp).isoformat(' ');
            year = tmstmp[0:4]; mon = tmstmp[0:7]; day = tmstmp[0:10]; hour = tmstmp[0:13];
            # insert an edge into the dictionary
            linkDict[(fromNode, toNode)] = tmstmp;
            # insert every tuple (fromNode, toNode) into the corresponding time period
            if year in yearDict:
                yearDict[year].append((fromNode, toNode));
            else:
                yearDict[year] = [(fromNode, toNode)];
                yearList.append(year);
            if mon in monDict:
                monDict[mon].append((fromNode, toNode));
            else:
                monDict[mon] = [(fromNode, toNode)];
                monList.append(mon);
            if day in dayDict:
                dayDict[day].append((fromNode, toNode));
            else:
                dayDict[day] = [(fromNode, toNode)];
                dayList.append(day);
            if hour in hourDict:
                hourDict[hour].append((fromNode, toNode));
            else:
                hourDict[hour] = [(fromNode, toNode)];
                hourList.append(hour);
    
    yearList.sort(); monList.sort(); dayList.sort(); hourList.sort();   # sort all these lists in an increasing order with respective to time
    fobj.close();
    
    