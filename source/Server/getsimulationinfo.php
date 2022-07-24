<?php
    require './helpers.php';

    $db = connectToDB();

    $response = $db->query("SELECT SIMULATION_ID as id, count(1) as likes FROM userlike GROUP BY SIMULATION_ID");

    $likesBySimulation = array();
    while($obj = $response->fetch_object()){
        $likesBySimulation[$obj->id] = (int)$obj->likes;
    }

    $response = $db->query(
        "SELECT 
            sim.ID as id, 
            sim.NAME as simulationName,
            sim.DESCRIPTION as description,
            u.NAME as userName,
            sim.WIDTH as width, 
            sim.HEIGHT as height, 
            sim.PARTICLES as particles,
            sim.VERSION as version, 
            sim.TIMESTAMP as timestamp,
            sim.NUM_DOWNLOADS as numDownloads,
            OCTET_LENGTH(sim.content) as contentSize
        FROM simulation sim
        LEFT JOIN
            user u
        ON
            u.ID=sim.USER_ID
        ");

    $result = array();
    while($obj = $response->fetch_object()){
        $likes = is_null($likesBySimulation[$obj->id]) ? 0 : $likesBySimulation[$obj->id];
        $result[] = [
            "id" => (int)$obj->id, 
            "simulationName" => htmlspecialchars($obj->simulationName), 
            "userName" => htmlspecialchars($obj->userName),
            "description" => htmlspecialchars($obj->description),
            "width" => (int)$obj->width,
            "height" => (int)$obj->height,
            "particles" => (int)$obj->particles,
            "version" => $obj->version,
            "timestamp" => $obj->timestamp,
            "contentSize" => $obj->contentSize,
            "likes" => $likes,
            "numDownloads" => (int)$obj->numDownloads
        ];
    }

    echo json_encode($result);
    $db->close();
?>
