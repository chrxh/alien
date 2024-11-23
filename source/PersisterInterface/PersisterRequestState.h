#pragma once

enum class PersisterRequestState
{
    InQueue,
    InProgress,
    Finished,
    Error
};
