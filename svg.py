from svgpathtools import parse_path

# <circle class="cls-1" cx="336.3" cy="188.16" r="0.08"/>
# <circle class="cls-1" cx="290.21" cy="281.87" r="0.05"/>
# <circle class="cls-1" cx="181.94" cy="304.01" r="0.05"/>
# <circle class="cls-1" cx="111.79" cy="351.98" r="0.05"/>
# <circle class="cls-1" cx="181.82" cy="219.63" r="0.05"/>
# <circle class="cls-1" cx="165.95" cy="94.86" r="0.05"/>

paths = [
"""M253,144.89a65.4,65.4,0,0,1,9.5-.54,38.52,38.52,0,0,1,8.87.91,25.33,25.33,0,0,1,6.79,2.44A24.93,24.93,0,0,1,284.8,153a23,23,0,0,1,4.85,7c1,2.31.6,2.69,2.58,9.19.62,2.06,1,3,1.2,3.67a44.73,44.73,0,0,0,2.76,6.08c1.4,2.54,2.1,3.81,3.12,4.76,2.21,2,4.81,2.4,8.55,2.92,1.17.16.81,0,10.05.41,1.69.07,2.31.1,3.46.13,4.33.11,4.86-.06,6.87.06a40.76,40.76,0,0,1,7.54,1.28,53.45,53.45,0,0,1,6.64,1.86,32.55,32.55,0,0,1,6.82,3.15,30.46,30.46,0,0,1,5.08,4.06,59.45,59.45,0,0,1,5.5,6.18,74.21,74.21,0,0,1,5.27,6.82A33.16,33.16,0,0,1,369,218c.67,1.92.49,2.07,1.55,5.73.52,1.79,1.13,3.89,2.12,6.44a39.25,39.25,0,0,0,4.29,8.69c.18.26,1.11,1.49,3,3.94,2.25,3,2.52,3.3,2.84,3.64.94,1,1.94,1.76,5.42,3.68a66.43,66.43,0,0,0,8.49,4.24,60.27,60.27,0,0,0,8,2.28c1.48.35,4.72,1.1,8.95,1.78,4.68.75,8.28,1.05,9.33,1.14,1.35.1,5,.37,10,.34a106.77,106.77,0,0,0,20.44-1.82,51.94,51.94,0,0,0,9.17-2.39c3.44-1.29,6.87-2.57,10.16-5.5,2.42-2.15,1.33-2.14,6.94-8.79,2.2-2.62,3.93-4.47,5.35-5.92,2.22-2.27,2.35-2.19,3.83-3.71,4.82-5,4.53-7,8.68-10.66a21,21,0,0,1,5.38-3.64,26.63,26.63,0,0,1,6.79-1.89,17.71,17.71,0,0,1,4.43-.5c.75.05,2.66.26,6.94,2.7a46.69,46.69,0,0,1,7.24,5c2.54,2.16,2.4,2.49,6,5.8,1.39,1.28,3.24,3,5.73,4.89,1.58,1.22,3.5,2.48,7.28,5,1.67,1.09,4.35,2.84,7.81,5.23,2.7,1.86,3.53,2.5,4.47,3.64a23.93,23.93,0,0,1,3,5.38c1.31,2.82,2,4.26,2,5.8a14.07,14.07,0,0,1-.8,4.63,34.11,34.11,0,0,1-2.16,5.76,42.61,42.61,0,0,1-4.06,6.71c-1.25,1.7-1.69,2-2.73,3.68a32.56,32.56,0,0,0-2.27,4.32c-1.91,4.3-1.28,4.82-2.85,8.15-1.63,3.5-2.37,3-3.18,5.73-.74,2.46-.35,3.58-1.59,5.8a29.88,29.88,0,0,0-1.75,2.92c-.78,1.66-.72,2.27-1.21,3.33-.77,1.67-1.89,2.28-3.79,3.79a64.6,64.6,0,0,0-5.8,5.54c-2.77,2.82-2.12,2.65-5.16,5.84-3.59,3.77-4.52,4.05-5.91,6.48-.7,1.22-1.36,2.7-2.16,4.51-.69,1.54-1.24,2.8-1.67,4s-.77,2.61-1.52,10.66c-.26,2.83-.39,4.27-.45,5.68s-.15,3.65,0,6.6c.2,4.08.61,4.18.53,6.82a41.32,41.32,0,0,1-.84,6.07,43.67,43.67,0,0,1-1.32,5.88c-.13.41-.73,1.91-1.94,4.92a41.4,41.4,0,0,1-2.39,5.39,15.31,15.31,0,0,1-1.59,2.54c-1.35,1.64-2.13,1.61-5,4a10.19,10.19,0,0,0-2.69,2.73,18.89,18.89,0,0,0-1.18,2.62,17.48,17.48,0,0,1-.91,2.35c-.81,1.3-1.54,1.17-2.46,2.38-.77,1-.51,1.44-1.23,3.13a12.26,12.26,0,0,1-2.37,3.74,11.37,11.37,0,0,1-4.55,2.84,20.67,20.67,0,0,1-4.44.8c-3.34.4-3.89.14-5.8.68a14.25,14.25,0,0,0-3.41,1.44,20.18,20.18,0,0,0-4.06,3.49,26,26,0,0,0-3.41,3.75,13.55,13.55,0,0,1-1.9,2.43c-.93.93-1.12.81-1.47,1.29-.7.93-.66,2.3.53,6.29.49,1.65.86,2.72,1.55,4.7,1.29,3.72,1.94,5.6,2.2,6.48.2.68.65,2.67,1.55,6.64.67,2.94.75,3.32.74,4.05a13.25,13.25,0,0,1-1.42,5.58,16.32,16.32,0,0,1-1.86,3.11c-1.69,2.21-3.8,3.5-8,5.46-5,2.34-9.26,3.9-11.07,4.55-3.85,1.38-4.84,1.55-7.88,2.84-2.19.93-3.74,1.71-6.49,3.11-4.8,2.44-7.2,3.67-8.83,4.74a39.52,39.52,0,0,0-9.21,7.85,23.94,23.94,0,0,1-3.68,4.09,13.17,13.17,0,0,1-6.48,3.19c-2.58.37-4.52-.36-7.6-1.51a39.07,39.07,0,0,1-7.27-3.8c-4.1-2.56-3.71-2.93-6.29-4.13
s-4.92-2.3-7.62-2.13a5.81,5.81,0,0,1-2.05-.11c-1.65-.43-2.26-1.49-3.48-2.58-2.21-1.95-3.1-1.05-7.7-3.79a19,19,0,0,1-6.14-5.5c-3.34-4.13-3.17-4.54-4.55-5.46a14.9,14.9,0,0,0-4.38-1.66,23,23,0,0,0-6.55-1.13,21,21,0,0,0-6.05.87,15.88,15.88,0,0,0-6.32,2.82,16.14,16.14,0,0,1-3.61,2.77c-1.2.59-1.48.45-2.43,1-1.43.82-2,1.82-3.28,3.59-1.4,1.94-.68.7-3.86,4.84a50.58,50.58,0,0,0-3.67,5.07,20.94,20.94,0,0,0-2.17,5,22,22,0,0,1-.9,3.46c-.63,1.52-1.06,1.63-2.1,3.48a33.45,33.45,0,0,0-1.84,4.23c-.76,1.92-1.12,3-1.64,4.37-1.79,4.86-2.69,7.29-3.79,9.09-1.31,2.14-2,2.6-2.51,4.38-.26,1-.35,1.92-1,3.82-.16.49-.36,1.05-.64,1.84,0,0-.46,1.31-.92,2.53-1,2.66-1.57,3.9-2.15,5.48-1.52,4.13-.77,4.33-1.95,6.89-.74,1.62-1.78,2.87-3.86,5.35a74.24,74.24,0,0,1-6.07,6.27,33.59,33.59,0,0,1-4.1,3.54,13.24,13.24,0,0,1-4,2.25c-1.67.52-2.17.19-4.15.87a18,18,0,0,0-2.91,1.36c-1.93,1-2.16,1.43-3.36,2.07-1.4.75-1.89.65-5.27,1.51-3.56.91-3.32,1.1-5.48,1.56a57.14,57.14,0,0,1-6.27.72c-4.62.54-4.2,1-8,1.33-4,.41-4.84,0-7.7.74-1.91.51-2,.82-4.28,1.52-3.37,1-4.4.72-7.12,1.33-2.29.51-2.17.86-10.29,4.6a57.94,57.94,0,0,1-6.09,2.62,54,54,0,0,1-8.3,1.56c-2.35.33-3.53.49-4.89.59-3.83.26-4.78-.19-10.65-.31-4.87-.1-6.33.16-7.7.61a16,16,0,0,0-4.28,2.18,47.43,47.43,0,0,0-4.25,4.48c-1.46,1.62-2.09,2.36-2.33,3.59a5.43,5.43,0,0,0,.1,2.45c.16.58.55,1.63,2.38,3.38s2.28,1.49,4,3.07a28.83,28.83,0,0,1,2.92,3.49c.55.7,1.36,1.8,2.36,3.27,1.33,2,1.73,2.79,2.89,4.66a62.64,62.64,0,0,0,4.71,6.87c1.7,2.1,2,2.08,3.84,4.48a43.07,43.07,0,0,1,3,4.25c1.57,2.58,2.35,3.87,2.59,5.35a16,16,0,0,0,.64,3.81c.21.6.34.81.79,2,0,0,.51,1.34,1,2.71a25.35,25.35,0,0,1,1.3,5.51,21.19,21.19,0,0,1-.2,5.71c-.35,2.07-.62,1.82-.87,3.61s0,2.29-.08,6.22c-.05,2.59-.18,3.09-.54,3.61-.6.87-1.3,1-2.4,2.25a10.84,10.84,0,0,0-1.23,1.84,13.62,13.62,0,0,1-5.07,5,11.68,11.68,0,0,1-7.37.69,16.32,16.32,0,0,1-4.2-1.36c-2.75-1.23-2.18-1.73-5.3-3.3a5.9,5.9,0,0,1-2.64-2.08,3.05,3.05,0,0,1-.54-1.56,6,6,0,0,1,.49-1.89,6.29,6.29,0,0,0,.41-2c0-.41-.12-.57-1-2.69l-.54-1.23a10.39,10.39,0,0,1-.64-3c-.19-1.8-1-3.22-2.15-5.48-.92-1.73-1-1.46-2.61-4-1.38-2.27-1.24-2.41-2.54-4.56s-2.43-3.7-3.3-4.89a34.85,34.85,0,0,1-3.07-4.32,23.32,23.32,0,0,1-1.56-3.64c-.6-1.78-.53-2.27-1.18-4.22-.11-.34-.34-1-1.28-3.2-1.09-2.56-1.64-3.83-2.33-5.2a50.53,50.53,0,0,0-3.51-5.76c-1.17-1.77-2.37-3.42-4.73-6.66-2-2.78-3-4.17-4.56-6.17-3-3.93-3-3.78-5-6.55a28.56,28.56,0,0,0-2.73-3.61c-1.87-1.91-2.8-2-4.64-4a18.6,18.6,0,0,1-2.56-3.54,23.87,23.87,0,0,1-2.38-6c-.15-.54-.45-2.05-1-5.07a57.2,57.2,0,0,1-1-6.17,16.43,16.43,0,0,1-.11-3.61,14,14,0,0,1,1.21-4.07c.65-1.5.88-1.46,1.79-3.4.63-1.35.58-1.47,1.3-3.13a38.11,38.11,0,0,1,1.87-3.76c.35-.6,1.17-1.71,2.79-3.94.86-1.18,1.65-2.41,2.54-3.56a7.41,7.41,0,0,0,1.25-2,7.76,7.76,0,0,0,.46-2.69c.1-1.79-.07-2.34,0-3.48a15.81,15.81,0,0,1,.87-4.12,37.73,37.73,0,0,0,.8-6.28c.52-6.29,0-6.12.43-10.7a53.35,53.35,0,0,1,1.72-9.73c.23-.79.54-1.77.87-3.3.54-2.52.46-3.49.82-5.66a35.27,35.27,0,0,1,1.48-5.73,68.18,68.18,0,0,1,2.67-6.51
c1.68-3.79,1.7-3.46,2.12-4.76,1.05-3.26.4-3.75,1.56-6.37a15.19,15.19,0,0,0,1.36-3.59c.28-1.32.09-1.53.28-2.51.25-1.26.82-2.2,3.2-5,1.81-2.15,2.71-3.22,3.38-3.85s2.4-2.15,7.5-4.73c3.47-1.76,4.2-1.84,8.81-4,4.09-1.91,3.82-2,6.63-3.27a95.39,95.39,0,0,1,12.42-4.59,97.6,97.6,0,0,0,10.39-3.84,7.32,7.32,0,0,0,3.31-2.56,9.06,9.06,0,0,0,1.17-3.73c.42-2.59.1-3.33.31-6.79.1-1.73.27-3,.49-4.63a48.25,48.25,0,0,1,.94-5.66,30,30,0,0,1,2.56-6.61,23.65,23.65,0,0,1,4.84-6.65c1.52-1.41,1.65-1.07,7.22-5.2a12.23,12.23,0,0,0,3.33-3.2,8.59,8.59,0,0,0,1.64-4.38,8.5,8.5,0,0,0-2.13-5.86,10,10,0,0,0-4.53-2.95,15.69,15.69,0,0,0-4.91-.79c-5.16-.32-7.74-.48-9.86-.72a42,42,0,0,1-7.78-1.43c-4.65-1.36-4.51-2.33-8.61-3.33a43,43,0,0,0-8.21-1c-2.22-.12-1.44,0-5.58-.08-4.53-.12-5.51-.31-7.17.33-1.28.5-1.41.88-3.28,1.57a38.16,38.16,0,0,1-3.79,1.07c-1.09.27-1.17.27-2.64.62-3.41.81-3.12.87-3.84.94a9.5,9.5,0,0,1-5.19-1,8.49,8.49,0,0,1-3-2.35,11.34,11.34,0,0,1-1.74-3.38,11.06,11.06,0,0,0-.79-2.3c-.54-.94-.89-1-1.59-2.05a19.34,19.34,0,0,1-1.1-2.23,19.83,19.83,0,0,0-1.72-2.66,17.93,17.93,0,0,1-2.87-5.51,13.43,13.43,0,0,1-.25-5,12.46,12.46,0,0,1,1.15-4.6,9.36,9.36,0,0,1,1.74-2.59,10.13,10.13,0,0,1,4.69-2.53c3.21-1,4.54-.93,8.45-1.69,2.08-.41,1.28-.35,6.52-1.57s6.64-1.4,8.35-1.25a14.24,14.24,0,0,1,4.84,1.2c1.34.61,1.47.94,4.25,2.72a21.16,21.16,0,0,0,4.68,2.58,8.25,8.25,0,0,0,5.1.26c1-.34,2.13-1,5.35-5.76,1.66-2.44,1.66-2.69,3.25-5.12,1.88-2.87,2-2.76,5.66-7.89,2.54-3.59,2.39-3.55,3.56-5s2-2.45,3.25-4.31,1.88-2.87,2-4.06c.15-1.41-.37-1.7-.5-4.23a9.09,9.09,0,0,1,.17-3.32c.64-2,1.85-1.87,2.48-3.88.35-1.11,0-1.24.42-3.64a24.78,24.78,0,0,1,.57-2.48,19.35,19.35,0,0,1,1.76-4.53,12.43,12.43,0,0,0,1.37-2.36c.31-.94.21-1.36.6-2.6s.55-1.08.8-1.94c.51-1.68-.12-2.23.21-4.06.25-1.36.63-1.17,1-2.63a9.25,9.25,0,0,0-.06-4.2c-.5-2-1.23-2-1.58-4-.25-1.4.08-1.55-.27-3-.26-1-.51-1.25-1.11-2.92-.34-1-.31-1.05-.74-2.45-.77-2.47-1-2.59-1.38-4a21.23,21.23,0,0,1-.68-3.94c-.23-2.45.11-2.68-.09-4.72a31.83,31.83,0,0,0-1.14-4.83c-.74-2.91-1.11-4.37-1.25-5.26-.47-3,.07-3.34-.3-7.31-.26-2.8-.52-2.57-.71-5.25-.17-2.35,0-2.7-.12-6.12-.09-2.16-.18-2.3-.3-4.81-.1-2-.19-3.75-.15-5.91a65.74,65.74,0,0,1,.8-8.83,18.84,18.84,0,0,1,.87-4.18c.62-1.62,1.06-1.83,2.57-4.6,1-1.89.86-1.87,2.12-4.24,1.42-2.67,1.51-2.48,2.12-3.85,1-2.32.7-2.79,1.67-4.83a14.18,14.18,0,0,1,2.33-3.59,18.12,18.12,0,0,1,5.19-3.7,26.87,26.87,0,0,1,4-1.91c2-.74,2.22-.52,3.73-1.16a17.15,17.15,0,0,0,4.3-2.75c.69-.55,1.66-1.52,3.61-3.46a20.24,20.24,0,0,0,2.57-2.78c.76-1.12.84-1.64,1.82-3.1a24.59,24.59,0,0,1,1.67-2.18c1.77-2,4.17-3,6.8-4.12a42.86,42.86,0,0,1,8.18-2.36C250.87,145.18,252.18,145,253,144.89Z""",
]

depths = [
    0
]

if __name__ == '__main__':
    for p, d in zip(paths, depths):
        x = parse_path(p)
        print(x)

# <svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 458.43 539.91"><defs><style>.cls-1{fill:none;stroke:#000;stroke-miterlimit:10;}</style></defs>
#         <path class="cls-1" d= transform="translate(-106.71 -143.84)"/>
#         </svg>

