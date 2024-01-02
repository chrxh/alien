<?php
    require './helpers.php';

    $db = connectToDB();
    $response = $db->query("SELECT SIMULATION_ID as id, TYPE as likeType, count(1) as likes FROM userlike GROUP BY SIMULATION_ID, TYPE");

    $likesBySimulationByType = array();
    while($obj = $response->fetch_object()){
        $likeType = is_null($obj->likeType) ? 0 : (int)$obj->likeType;

        if (!isset($likesBySimulationByType[$obj->id])) {
            $likesBySimulationByType[$obj->id] = array();
        }
        if (array_key_exists($likeType, $likesBySimulationByType[$obj->id])) {
            $likesBySimulationByType[$obj->id][$likeType] += (int)$obj->likes;
        }
        else {
            $likesBySimulationByType[$obj->id][$likeType] = (int)$obj->likes;
        }
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
            sim.SIZE as contentSize,
            sim.FROM_RELEASE as workspace,
            sim.TYPE as type
        FROM simulation sim
        LEFT JOIN
            user u
        ON
            u.ID=sim.USER_ID
        ");

    $userName = array_key_exists('userName', $_POST) ? $_POST['userName'] : '';
    $pw = array_key_exists('password', $_POST) ? $_POST['password'] : '';
    $pwCorrect = checkPw($db, $userName, $pw);

    $result = array();
    while($obj = $response->fetch_object()){
        $totalLikes = 0;

        if ((int)$obj->workspace == 2 && (!$pwCorrect || strcmp($obj->userName, $userName) != 0)) {
            continue;
        }
        $likesByType = array();
        if (isset($likesBySimulationByType[$obj->id])) {
            foreach($likesBySimulationByType[$obj->id] as $likeType => $likes) {
                $totalLikes += $likes;
                $likeType2 = is_null($likeType) ? 0 : $likeType;
                $likesByType[$likeType2] = $likes;
            }
        }
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
            "likes" => $totalLikes,
            "likesByType" => $likesByType,
            "numDownloads" => (int)$obj->numDownloads,
            "fromRelease" => (int)$obj->workspace,
            "type" => is_null($obj->type) ? 0 : $obj->type
        ];
    }

    echo json_encode($result);
    $db->close();
?>
