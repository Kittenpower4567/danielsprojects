async def complex_map(scale,gridenabled,ringenabled,xoff,yoff,res,cinput):
    # Volba is the goat for making the speed go from hours to minutes (or seconds in some cases)
    cinput = str(cinput).replace('^','**')
    def f(z):
        # Define f(z) here
        return eval(cinput)
    gridenabled = gridenabled
    ringenabled = ringenabled
    # Canvas scale
    scale = float(scale)
    xoff = float(xoff)
    yoff = float(yoff)
    res = int(res)
    if res < 16:
        await message.channel.send("You cannot have images smaller than 16px")
        res = 16
    if res > 1000:
        await message.channel.send("You cannot have images larger than 1000px")
        res = 1000
    await message.channel.send('''🔁 Output is being created...
Please note that High image resolutions or large operations can take longer.''')
    start = time.time()

    # Define function to map a complex number to an RGB value
    def ringbool(z):
        if ringenabled.lower() == "true" or ringenabled.lower() == "t":
            return (np.clip((abs(16*z)-16)**2, 0, 1)**100000)
        else:
            return 1
    def grid(z):
        if gridenabled.lower() == "true" or gridenabled.lower() == "t":
            return np.abs(np.sign(np.sin(np.pi * np.real(16*z))) + np.sign(np.sin(np.pi * np.imag(16*z))))
        else:
            return 0
    def ring(z):
        if ringenabled.lower() == "true" or ringenabled.lower() == "t":
            return (np.clip((abs(16*z)-16)**2, 0, 1)**100000)
        else:
            return 1
    def otherring(z):
        if ringenabled.lower() == "true" or ringenabled.lower() == "t":
            return np.clip((z**-0.3)*(np.clip((abs(16*z)-16)**2,0,1)**100000),0,1)+((1-np.clip((abs(16*z)-16)**2,0,1)**100000)/1.5)
        else:
            return np.clip(z**-0.3,0,1)
    def color(z):
        r, theta = np.abs(z), np.angle(z)
        h = np.fmod((np.arctan2(-z.imag, z.real) / np.pi + 1) / 2 + 0.5, 1)
        v = otherring(r)
        s = np.clip(((r**0.3)*0.8)+0.2, 0.2, 1)*ring(r) # adjust saturation based on absolute value
        # Create a zero-filled UInt8 array with 3 values per pixel
        color_array = np.zeros(z.shape + (3,), dtype=np.uint8)
        # Create a boolean mask for grid filtering
        gridmask = grid(z)*ringbool(z) == 0 
        # Assign new values only where the mask is True
        color_array[gridmask] = np_hsv_to_rgb(h, s, v)[gridmask]
        return color_array
    # Define function to convert HSV to RGB
    def hsv_to_rgb(h, s, v):
        if s == 0.0:
            return v, v, v

        i = int(h * 6.0)
        f = (h * 6.0) - i
        p, q, t = v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))

        i %= 6
        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q


    # Define vectorized HSV to RGB function
    def np_hsv_to_rgb(h, s, v):
        def hue_trapezoid(angle):
            sawtooth = np.abs(angle % 1 - 0.5) * 6 - 1
            return np.clip(sawtooth, 0, 1)

        # Give each value a new, empty axis (to fit the rgb calculations)
        h2, s2, v2 = h[:,:,None], s[:,:,None], v[:,:,None]
        # Three offsets mapping the hue to each rgb channel
        corners = np.array([0, 1/3, 2/3])[None,None,:]

        normalized_rgb = ((hue_trapezoid(h2 - corners) - 1) * s2 + 1) * v2
        return np.round(255 * normalized_rgb)


    # Set image size
    width, height = res, res
    # img = Image.new('RGB', (width, height), color=(0, 0, 0))
    xmin, xmax = -scale, scale
    ymin, ymax = -scale, scale


    # 2D array mapping pixel coordinates to complex numbers
    z = np.array([
        [complex(xmin + (xmax - xmin) * x / (width - 1),
                ymin + (ymax - ymin) * y / (height - 1))
        for x in range(width)]
        for y in range(height)
    ])

    # Apply function f(z)
    w = f(z+complex(xoff,yoff))
    # Map new complex number to RGB color
    col = color(w)
    # Interpret array as color values
    img = Image.fromarray(col)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.0)
    # Save image to file
    file = 'output.png'
    img.save(file)
    end = time.time()
    await message.channel.send("Image created. Operation took " + str(np.floor(1000.0*(end-start))) + "ms.")
    await send_image('output.png')

# Split the message content to extract arguments
args = message.content.split(' ')[1:]
Ar = ((((' '.join([str(elem) for elem in args])).replace(']','')).replace('[',''))).split()
#wait message.channel.send(Ar[0].join(Ar[1]).join(Ar[2]).join(Ar[3]).join(Ar[4]).join(Ar[5]))
if len(args) >= 7:
    scale = Ar[0].replace(',','')
    gridenabled = Ar[1].replace(',','')
    ringenabled = Ar[2].replace(',','')
    xoff = Ar[3].replace(',','')
    yoff = Ar[4].replace(',','')
    res = Ar[5].replace(',','')
    cinput = Ar[6] 
    await complex_map(scale, gridenabled, ringenabled, xoff, yoff, res, cinput)
