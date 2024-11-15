def twoStrings(s1, s2):
    # Write your code here
    totl=len(set(s1))+len(set(s2))
    ddoup=len(set(s1+s2))
    retval="NO"
    if totl != ddoup:
        retval="YES" 
    return retval
