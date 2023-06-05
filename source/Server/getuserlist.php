<?php
    require './helpers.php';

    $db = connectToDB();

    $response = $db->query(
        "SELECT 
            u.NAME as userName,
            u.TIMESTAMP as timestamp
        FROM user u");

    $result = array();
    while($obj = $response->fetch_object()){
        $likes = is_null($likesBySimulation[$obj->id]) ? 0 : $likesBySimulation[$obj->id];
        $result[] = [
            "userName" => htmlspecialchars($obj->userName),
            "starsEarned" => 0,
            "starsGiven" => 0,
            "timestamp" => $obj->timestamp
        ];
    }

    echo json_encode($result);
    $db->close();
?>
