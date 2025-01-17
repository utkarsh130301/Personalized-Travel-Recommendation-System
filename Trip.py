def read_data_to_trips(data, DEBUG=0) :
  if DEBUG>0:
    print ("### read_data_to_trips( [{}] )".format(data.shape))
  trips = []
  aTrip = None
  [rows,cols] = data.shape

  seqID=last_seqid=-1
  
  for index, row in data.iterrows():
  #for r in range(rows) :
    last_seqid=seqID

    userid = row['userID']
    time = row['dateTaken']
    seqID = row['seqID']
    poiId = row['poiID']
    #imgURL = img=data['IMGURL']

    ### check new trip?
    if aTrip == None :
      aTrip = Trip(userid)
      aTrip.add(time, poiId)
      trips.append(aTrip)

    elif aTrip.userid == userid and seqID==last_seqid:
      ## CONTINUE TRIP
      aTrip.add(time, poiId)

    elif aTrip.userid!=userid or seqID!=last_seqid:
      ## NEW TRIP
      # print("\nL392 starting new trip  (", userid ,")");

      aTrip = Trip(userid)
      trips.append(aTrip)

    else:
      print ("SIZE", aTrip.size())
      quit(0)
      for s in range( aTrip.size() ):
        print (aTrip.userid)
        print (aTrip.getStep(s))
        aTrip.add(time, lat, lng, woeid, imgURL)

  for trip in trips:
    days = []
    today = []
    last_poi, last_time = None, None
    return trips

    utrip = []
    print ("###  L186 -- read_data_to_trips(...) :  [",len(trips),"]")
    for (t,poiid) in trip.items:
      if len(utrip) == 0:
        utrip.append((t,poiid))
        lastpoi = poiid
      elif poiid != utrip[-1][1]:
        utrip.append((t,poiid))
        lastpoi = poiid
    print ("###  L175 -- read_data_to_trips(...) :  [",len(trips),"]")
  return trips
