def checkMagazine(magazine, note):
         # Write your code here
    nt_wrds=len(set(note))
    nt_mag=len(set(magazine))
    mix_wrds= len(set((note+ magazine)))
    ret="No"
    if nt_mag==mix_wrds:
        ret="Yes"
    return ret
