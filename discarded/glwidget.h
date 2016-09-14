#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>

class GLWidget : public QGLWidget
{
Q_OBJECT

 public:
   GLWidget( int timerInterval=0, QWidget *parent=0, char *name=0 );
   ~GLWidget ();

 protected:
   virtual void initializeGL();
   virtual void resizeGL( int width, int height );
   virtual void paintGL();


   GLfloat rtri;
   GLfloat rquad;

//   virtual void keyPressEvent( QKeyEvent *e );

 protected slots:
   virtual void timeOutSlot();

 private:
   QTimer *m_timer;
};

#endif // GLWIDGET_H
