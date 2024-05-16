import pygame

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

def init_system(): 
    pygame.init()

class GUI: 
    def __init__(self, width, height, fps=30): 
        
        self.window_width = width
        self.window_height = height
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Music Production")

        self.event_components = []
        self.event_processors = []
        self.idle_components = [] 
        self.circles = []
        self.per_frame_call = []
        self.text_boxes = []
        self.active_text_box = -1
        self.running = False

        self.font = pygame.font.Font(None, 36)

        self.fps = fps
    
    def add_button(self, text, width_frac, height_frac, center_frac_x, center_frac_y, callback=None): 
        rect = pygame.Rect(
            center_frac_x * self.window_width, 
            center_frac_y * self.window_height, 
            width_frac * self.window_width, 
            height_frac * self.window_height
        )

        self.event_components.append((text, rect, callback)) 
        return len(self.event_components) - 1

    
    def event_trigger(self, event): 
        if event.type == pygame.QUIT: 
            self.running = False 
            return True

        for text, component, cb in self.event_components: 
            if not event.type == pygame.MOUSEBUTTONDOWN: 
                continue
            if component.collidepoint(event.pos) and cb is not None: 
                cb()
        
        self.active_text_box = -1
        for i in range(len(self.text_boxes)): 
            text, component, active = self.text_boxes[i] 
            if not event.type == pygame.MOUSEBUTTONDOWN: 
                continue

            self.text_boxes[i] = text, component, component.collidepoint(event.pos)
            if component.collidepoint(event.pos): 
                self.active_text_box = i

        for i in range(len(self.text_boxes)):
            text, component, active = self.text_boxes[i]
            if not active: 
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    active = False
                    self.active_text_box = -1
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += event.unicode
                self.text_boxes[i] = text, component, active 
            

        for processor in self.event_processors: 
            processor(event)
        
        return False
            
    
    def add_text(self, text, center_frac_x, center_frac_y): 
        text = self.font.render(text, True, black)
        text_rect = text.get_rect(center=(center_frac_x * self.window_width, center_frac_y * self.window_height))

        self.idle_components.append((text, text_rect))
    
    def add_circle(self, radius, center_frac_x, center_frac_y): 
        self.circles.append(((center_frac_x * self.window_width, center_frac_y * self.window_height), radius, black))
    
    def add_textbox(self, width_frac, height_frac, center_frac_x, center_frac_y): 
        rect = pygame.Rect(
            center_frac_x * self.window_width, 
            center_frac_y * self.window_height, 
            width_frac * self.window_width, 
            height_frac * self.window_height
        )

        self.text_boxes.append(('', rect, False)) 

    def get_text(self): 
        ret = []
        for text, _, _ in self.text_boxes: 
            ret.append(text)

        return ret 
    
    def reset(self): 
        self.window.fill(white)
        
    
    def start(self): 
        self.running = True 
    
    def add_per_frame(self, func): 
        self.per_frame_call.append(func)
    
    def add_event_processor(self, func): 
        self.event_processors.append(func)

    def render(self): 
        
        for func in self.per_frame_call: 
            func()

        for text, component, cb in self.event_components: 
            pygame.draw.rect(self.window, black, component, 2)
            render_text = self.font.render(text, True, black)
            text_rect = render_text.get_rect(center=component.center) 
            self.window.blit(render_text, text_rect)
        

        for text, component, active in self.text_boxes: 
            color = (red if active else black)
            pygame.draw.rect(self.window, color, component, 3)
            if text == '' and not active: 
                text = 'Enter Text Here'
            render_text = self.font.render(text, True, black)
            text_rect = render_text.get_rect(center=component.center) 
            self.window.blit(render_text, text_rect)

        for text, text_rect in self.idle_components: 
            self.window.blit(text, text_rect)

        for center, radius, color in self.circles: 
            pygame.draw.circle(self.window, color, center, radius, width=0)


        pygame.display.flip()


# pygame.init() 
# gui = GUI(width=800, height=600) 

# gui.add_button(
#     "Start Production", 
#     0.4, 0.0625, 0.3, 0.4, 
#     callback=lambda: print("START PRODUCTION")
# ) 

# gui.add_button(
#     "See Result", 
#     0.4, 0.0625, 0.3, 0.65, 
#     callback=lambda: print("See Results")
# ) 

# gui.add_text(
#     text="Are you ready to produce your music?", 
#     center_frac_x=0.5, 
#     center_frac_y = 0.25
# )
# running = True 


# gui2 = GUI(width=800, height=600) 

# gui2.add_button(
#     "Start Production 2", 
#     0.4, 0.0625, 0.3, 0.4, 
#     callback=lambda: print("START PRODUCTION 2")
# ) 

# gui2.add_button(
#     "See Result 2", 
#     0.4, 0.0625, 0.3, 0.65, 
#     callback=lambda: print("See Results 2")
# ) 

# gui2.add_text(
#     text="Are you ready to produce your music? 2", 
#     center_frac_x=0.5, 
#     center_frac_y = 0.25
# )
# running = True 


# i = 0 
# while running: 
#     if i < 100: 
#         g = gui 
#     else: 
#         g = gui2 
    
#     for event in pygame.event.get(): 
#         g.event_trigger(event) 
    
#     g.reset() 
#     g.render()
#     i += 1



        
        
        
        
    