#pragma once

#include "Web/Definitions.h"

#include "Definitions.h"

class BugReportController : public QObject
{
    Q_OBJECT
public:
    BugReportController(std::string const& errorMessage, std::string const& protocol);
    ~BugReportController();

    void execute();

private:
    BugReportView* _view = nullptr;
    WebAccess* _webAccess = nullptr;
};
