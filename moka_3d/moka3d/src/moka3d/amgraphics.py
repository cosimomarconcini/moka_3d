import matplotlib
import matplotlib.pyplot as plt

def showplot(fig, xshift=0, yshift=0, pdffile='test.pdf', transparent=True):
    #print matplotlib.get_backend()
    if matplotlib.get_backend() == 'Qt5Agg':
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
        mngr.window.setGeometry(x+xshift, y+yshift, dx, dy)
        fig.canvas.manager.window.activateWindow()
        fig.canvas.manager.window.raise_()
        fig.show()
    elif matplotlib.get_backend() == 'pdf':
        print('saving to file: %s' % pdffile)
        fig.savefig(pdffile, transparent=transparent)
