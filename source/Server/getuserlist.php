<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $response = $db->query("SELECT u.ID as id, count(1) as likes FROM user u, simulation s, userlike ul WHERE s.USER_ID = u.ID AND ul.SIMULATION_ID = s.ID GROUP BY u.ID");
    $starsReceivedByUser = array();
    while($obj = $response->fetch_object()){
        $starsReceivedByUser[$obj->id] = (int)$obj->likes;
    }

    $response = $db->query("SELECT ul.USER_ID as id, count(1) as likes FROM userlike ul GROUP BY ul.USER_ID");
    $likesGivenByUser = array();
    while($obj = $response->fetch_object()){
        $likesGivenByUser[$obj->id] = (int)$obj->likes;
    }

    $response = $db->query("SELECT u.ID as id FROM user u WHERE u.TIMESTAMP >= DATE_SUB(NOW(), INTERVAL 60 MINUTE) AND u.FLAGS = 1");
    $onlineByUser = array();
    while($obj = $response->fetch_object()){
        $onlineByUser[$obj->id] = true;
    }

    $response = $db->query("SELECT u.ID as id FROM user u WHERE u.TIMESTAMP >= DATE_SUB(NOW(), INTERVAL 60*24 MINUTE)");
    $lastDayOnlineByUser = array();
    while($obj = $response->fetch_object()){
        $lastDayOnlineByUser[$obj->id] = true;
    }

    $response = $db->query(
        "SELECT 
            u.ID as id,
            u.NAME as userName,
            u.TIMESTAMP as timestamp,
            u.TIME_SPENT as timeSpent,
            u.GPU as gpu
        FROM
            user u
        WHERE
            u.ACTIVATION_CODE=''");
    $result = array();
    while($obj = $response->fetch_object()){
        $starsReceived = is_null($starsReceivedByUser[$obj->id]) ? 0 : $starsReceivedByUser[$obj->id];
        $likesGiven = is_null($likesGivenByUser[$obj->id]) ? 0 : $likesGivenByUser[$obj->id];
        $online = !is_null($onlineByUser[$obj->id]);
        $lastDayOnline = !is_null($lastDayOnlineByUser[$obj->id]);
        $timeSpent = !is_null($obj->timeSpent) ? (int)$obj->timeSpent : 0;
        $gpu = !is_null($obj->gpu) ? $obj->gpu : "";
        $result[] = [
            "userName" => htmlspecialchars($obj->userName),
            "starsReceived" => $starsReceived,
            "starsGiven" => $likesGiven,
            "timestamp" => $obj->timestamp,
            "online" => $online,
            "lastDayOnline" => $lastDayOnline,
            "timeSpent" => $timeSpent,
            "gpu" => $gpu
        ];
    }

    echo json_encode($result);
    $db->close();
?>
