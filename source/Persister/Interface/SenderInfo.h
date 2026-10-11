#pragma once

#include "SenderId.h"

struct SenderInfo
{
    SenderId senderId;
    bool wishResultData = true;
    bool wishErrorInfo = true;
};
