import os
csvs = os.listdir(".")
for csv in csvs:
    wekacsv = open(csv,"r")
    if "pythonCSV" not in csv:
        #print(csv[-12:-3])
        pythoncsv = open(csv[:-3]+"pythonCSV.csv","w")
        lines = wekacsv.readlines()
        linecount = 0
        for line in lines:
            if linecount is not 0:
                #print(line[-2:-1])
                if line[-2:-1] == 'y':
                    pythoncsv.write(line[:-2]+'1\n') 
                else:
                    pythoncsv.write(line[:-2]+'-1\n')
            linecount += 1
