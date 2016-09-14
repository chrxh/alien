#include "glwidget.h"

#include <QTimer>

GLWidget::GLWidget(int timerInterval, QWidget *parent, char *name)
    : QGLWidget( parent )
{
    m_timer = new QTimer();
    connect(m_timer, SIGNAL(timeout()), this, SLOT(timeOutSlot()));
    m_timer->start(10);
    rtri = 0.0f;
}
GLWidget::~GLWidget ()
{
    makeCurrent();
}

void GLWidget::initializeGL ()
{
    glShadeModel( GL_SMOOTH );
    glClearColor(0.0f, 0.0f, 0.0f, 0.5f);                                       // Let OpenGL clear to black
    glClearDepth(1.0f);									// Depth Buffer Setup
    glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
    glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations
}

void GLWidget::resizeGL( int width, int height )
{
    glViewport( 0, 0, (GLint)width, (GLint)height );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glFrustum( -1.0, 1.0, -1.0, 1.0, 5.0, 20.0 );
    glMatrixMode( GL_MODELVIEW );
}


void GLWidget::paintGL()
{
    rtri += 3.0f;

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    for(int j = 0; j < 300; ++j)
    for(int i = 0; i < 300; ++i) {
        glLoadIdentity();
        glTranslatef(-1.5f+(GLfloat)i/50.0f,1.5f-(GLfloat)j/50.0f,-15.0f);
        glRotatef(rtri,1.0f,0.3f,0.0f);
        glScalef(0.008f,0.008f,0.008f);
        glBegin(GL_TRIANGLES);						// Drawing Using Triangles
            glColor3f(1.0f,1.0f,0.0f);          // Red
            glVertex3f( 0.0f, 1.0f, 0.0f);          // Top Of Triangle (Front)
            glColor3f(0.0f,1.0f,0.0f);          // Green
            glVertex3f(-1.0f,-1.0f, 1.0f);          // Left Of Triangle (Front)
            glColor3f(0.0f,0.0f,1.0f);          // Blue
            glVertex3f( 1.0f,-1.0f, 1.0f);
            glColor3f(1.0f,1.0f,0.0f);          // Red
            glVertex3f( 0.0f, 1.0f, 0.0f);          // Top Of Triangle (Right)
            glColor3f(0.0f,0.0f,1.0f);          // Blue
            glVertex3f( 1.0f,-1.0f, 1.0f);          // Left Of Triangle (Right)
            glColor3f(0.0f,1.0f,0.0f);          // Green
            glVertex3f( 1.0f,-1.0f, -1.0f);         // Right Of Triangle (Right)
            glColor3f(1.0f,1.0f,0.0f);          // Red
            glVertex3f( 0.0f, 1.0f, 0.0f);          // Top Of Triangle (Back)
            glColor3f(0.0f,1.0f,0.0f);          // Green
            glVertex3f( 1.0f,-1.0f, -1.0f);         // Left Of Triangle (Back)
            glColor3f(0.0f,0.0f,1.0f);          // Blue
            glVertex3f(-1.0f,-1.0f, -1.0f);         // Right Of Triangle (Back)
            glColor3f(1.0f,1.0f,0.0f);          // Red
            glVertex3f( 0.0f, 1.0f, 0.0f);          // Top Of Triangle (Left)
            glColor3f(0.0f,0.0f,1.0f);          // Blue
            glVertex3f(-1.0f,-1.0f,-1.0f);          // Left Of Triangle (Left)
            glColor3f(0.0f,1.0f,0.0f);          // Green
            glVertex3f(-1.0f,-1.0f, 1.0f);          // Right Of Triangle (Left)
        glEnd();							// Finished Drawing The Triangle
    }

    glLoadIdentity();
    glTranslatef(1.5f,0.0f,-15.0f);					// Move Right 3 Units
    glRotatef(rtri,0.0f,1.0f,0.0f);

    glBegin(GL_QUADS);						// Draw A Quad
            glColor3f(1.0f,0.0f,0.0f);
            glVertex3f(-1.0f, 1.0f, 0.0f);				// Top Left
            glColor3f(0.0f,1.0f,0.0f);
            glVertex3f( 1.0f, 1.0f, 0.0f);				// Top Right
            glVertex3f( 1.0f,-1.0f, 0.0f);				// Bottom Right
            glVertex3f(-1.0f,-1.0f, 0.0f);				// Bottom Left
    glEnd();							// Done Drawing The Quad

}

void GLWidget::timeOutSlot()
{
    updateGL();
}
