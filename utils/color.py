TYPE_COLOR_MAP = {
    'C': '#ff6666',
    'O': 'lightskyblue',
    'N': 'yellowgreen',
    'movie': '#ff6666',
    'tvSeries': '#ff6666',
    'actor': 'lightskyblue',
    'actress': '#ffb3e6',
    'director': 'yellowgreen',
    'composer': '#c2c2f0',
    'producer': '#ffcc99',
    'cinematographer': 'gold'}
FAVORITE_COLORS = ['#ff6666', 'lightskyblue', 'yellowgreen', '#c2c2f0', 'gold',
                   '#ffb3e6', '#ffcc99', '#E0FFFF', '#7FFFD4', '#20B2AA',
                   '#FF8C00', '#ff1493',
                   '#FFE4B5', '#e6e6fa', '#7CFC00']

# SET_COLORS = ['#00B0F0', '#40B0FF', '#FF6666', '#FFD966', '#BEE396',
#               '#5DD4FF', '#0070C0', '#FF0000', '#FFC000', '#92D050',
#               '#0084B4', '#005490', '#BF0000', '#BF9000', '#6EAA2E']
SET_COLORS = ['#FFB4CD', '#A0DC78', '#78B9FF',
              '#FFDC64', '#BEA0FF', '#78DCA0',
              '#78A5DC', '#FFB978', '#78C8CD',
              '#FA8C8C', '#DC9CFA', '#8CC88C']

COLOR1 = ['#00B0F0']

COLOR2 = ['#CCCCCC']

COLOR3 = ["#EDEDBA"]

PURPLE_BLUE_DICT = {
    0: (190,160,255),   # #BEA0FF（紫藤紫）
    0.25: (160,130,240),# #A082F0（浅靛紫）
    0.5: (130,100,220), # #8266DC（中紫蓝）
    0.75: (100,70,200), # #6446C8（深紫蓝）
    1: (70,40,180)    # #4628B4（深蓝）
}

WARM_GRADIENT_DICT = {
    0: (230, 220, 210),    # #E6DCD2（低饱和暖灰）
    0.25: (240, 200, 170), # #F0C8AA（低饱和浅橙）
    0.5: (245, 170, 120),  # #F5AA78（中饱和橙）
    0.75: (250, 130, 70),  # #FA8246（高饱和橙红）
    1: (255, 90, 40)       # #FF5A28（高饱和鲜红）
}

CONTRAST_GRADIENT_DICT = {
    0: (200, 225, 220),     # #C8E1DC（低饱和青灰，冷色）
    0.25: (215, 210, 190),  # #D7D2BE（低饱和青褐，过渡色）
    0.5: (230, 190, 160),   # #E6BEA0（中饱和橙褐，中性色）
    0.75: (245, 150, 100),  # #F59664（高饱和橙红，暖色）
    1: (255, 90, 60)        # #FF5A3C（高饱和鲜红，强暖色）
}

ENHANCED_CONTRAST_DICT = {
    0: (180, 210, 230),     # #B4D2E6（低饱和青蓝，冷色基准）
    0.25: (210, 190, 180),  # #D2BEB4（低饱和橙灰，快速过渡）
    0.5: (230, 150, 120),   # #E69678（中饱和橙，完成过渡）
    0.75: (245, 110, 80),   # #F56E50（高饱和橙红，强化暖色）
    1: (255, 70, 50)        # #FF4632（高饱和鲜红，强暖色终点）
}

HIGH_CONTRAST_GRADIENT = {
    0: (120, 180, 255),     # #78B4FF（中饱和天蓝，冷色）
    0.25: (180, 160, 220),  # #B4A0E6（中饱和紫蓝，过渡色）
    0.5: (230, 130, 140),   # #E6828C（中饱和粉橙，完成过渡）
    0.75: (245, 90, 80),    # #F55A50（高饱和橙红）
    1: (255, 60, 40)        # #FF3C28（高饱和鲜红）
}

red_list = []
green_list = []
blue_list = []

for key in HIGH_CONTRAST_GRADIENT.keys():
    r, g, b = HIGH_CONTRAST_GRADIENT[key]
    # 将RGB值从0-255范围转换为0-1范围
    red_list.append((key, r/255, r/255))
    green_list.append((key, g/255, g/255))
    blue_list.append((key, b/255, b/255))

MAPDICT = {
    'red': tuple(red_list),
    'green': tuple(green_list),
    'blue': tuple(blue_list)
}