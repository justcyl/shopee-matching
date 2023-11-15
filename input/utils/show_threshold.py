def show(df,pairs):
    js=0
    mx=0
    for x,y,v in pairs:
        js+=1
        mx=max(mx,x)
        # print(x,y,v)
    print(mx)