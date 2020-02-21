
l=0
h=100
print("Please think of a number between 0 and 100!")
while(True):
    numb = (h+l) // 2 
    print("Is your secret number "+str(numb)+"?") 
    indication = input("Enter 'h' to indicate the guess is too high. Enter 'l' to indicate the guess is too low. Enter 'c' to indicate I guessed correctly. ")
    if indication not in "hlc": print("Sorry, I did not understand your input.")
    elif indication=="c":retval =numb; print("Game over. Your secret number was: "+str(numb)); break
    elif indication=="h": 
        h=numb
    elif indication=="l": 
        l=numb
        

