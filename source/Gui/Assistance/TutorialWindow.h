#ifndef TUTORIALWINDOW_H
#define TUTORIALWINDOW_H

#include <QMainWindow>

namespace Ui {
class TutorialWindow;
}

class TutorialWindow : public QMainWindow
{
    Q_OBJECT
    
public:
     TutorialWindow(QWidget *parent = 0);
    ~TutorialWindow();
    
Q_SIGNALS:
    void closed ();

protected:
    bool event(QEvent* event);

private:
    Ui::TutorialWindow *ui;
};

#endif // TUTORIALWINDOWS_H
