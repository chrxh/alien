#pragma once

#include <QDialog>

namespace Ui
{
    class BugReportDialog;
}

class BugReportView : public QDialog
{
    Q_OBJECT

public:
    BugReportView(std::string const& errorMessage, std::string const& protocol, QWidget* parent = nullptr);
    virtual ~BugReportView();

    std::string getProtocol() const;
    std::string getEmailAddress() const;
    std::string getUserMessage() const;

private:
    Ui::BugReportDialog* ui;
};
