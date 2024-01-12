import pygame
import pygame_widgets
from pygame_widgets.textbox import TextBox


class LevelSelect:
    def __init__(self, msg="Please enter the level you want to play (confirm with enter):",
                 px=800, py=200):
        self._running = None
        self._level_select = -1
        self._top_msg = msg
        self._tlist = []

        self._display_surf = None
        self.disp_size = (px, py)

    def execute(self):
        if not self.pygame_init():
            self._running = False

        while self._running:
            events = pygame.event.get()
            for event in events:
                self.on_event(event)
                self.on_loop()
                self.render()

            pygame_widgets.update(events)

        self.cleanup()

        return self._level_select

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        return event.type

    def on_loop(self):
        for elem in self._tlist:
            try:
                self._level_select = int(elem)
            except ValueError:
                # Ignore element
                pass
            else:
                self._running = False
                break

        self._tlist.clear()

    def pygame_init(self):
        try:
            # pygame.init()
            self._display_surf = pygame.display.set_mode(
                self.disp_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("Level Selection")
        except:
            # pygame.quit()
            return False

        w, h = self._display_surf.get_size()
        self._textbox = TextBox(self._display_surf, 100, 100, w-200, 80, fontSize=48,
                                borderColour="black", textColour="black",
                                onSubmit=lambda: self._tlist.append(self._textbox.getText()),
                                radius=0, borderThickness=5)

        self._running = True
        self.render()

        return True

    def cleanup(self):
        pass  # main program quits pygame

    def render(self):
        w, h = self._display_surf.get_size()
        self._display_surf.fill("white")

        font_size = 36
        font = pygame.font.SysFont(None, font_size)
        msg = font.render(self._top_msg, True, "black")
        msg_rect = msg.get_rect(center=(w / 2, 50))
        self._display_surf.blit(msg, msg_rect)

        self._textbox.draw()

        pygame.display.update()
