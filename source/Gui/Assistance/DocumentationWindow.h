#pragma once

#include <QMainWindow>

namespace Ui {
class DocumentationWindow;
}

class DocumentationWindow : public QMainWindow
{
    Q_OBJECT
    
public:
     DocumentationWindow(QWidget *parent = 0);
	 virtual ~DocumentationWindow();
    
Q_SIGNALS:
    void closed ();

protected:
    bool event(QEvent* event);

private:
    Ui::DocumentationWindow *ui;
};

