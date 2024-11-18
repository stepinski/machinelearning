def countTriplets(arr,r):
    trips_3={}
    trips_2={}
    trips=0

    for a in arr:
        if a in trips_3:
            trips+=trips_3[a]

        if a in trips_2:
            if a*r in trips_3:
                trips_3[a*r]+=trips_2[a]
            else:
                trips_3[a*r]=trips_2[a] 
        if a*r in trips_2:
            trips_2[a*r]+=1
        else:
            trips_2[a*r]=1
    return trips
