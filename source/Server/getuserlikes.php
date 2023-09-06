<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $simId = $_POST["simId"];
    if (array_key_exists("likeType", $_POST)) {
        $likeType = $_POST["likeType"];
        if ((int)$likeType == 0) {
            $response = $db->query(
                "SELECT 
                    u.NAME as userName
                FROM
                    userlike ul, user u
                WHERE
                    ul.USER_ID = u.ID
                    AND ul.SIMULATION_ID=$simId
                    AND (ul.TYPE=$likeType OR ul.TYPE IS NULL) ");
        }
        else {
            $response = $db->query(
                "SELECT 
                    u.NAME as userName
                FROM
                    userlike ul, user u
                WHERE
                    ul.USER_ID = u.ID
                    AND ul.SIMULATION_ID=$simId
                    AND ul.TYPE=$likeType");
        }
    }
    else {
        $response = $db->query(
            "SELECT 
                u.NAME as userName
            FROM
                userlike ul, user u
            WHERE
                ul.USER_ID = u.ID
                AND ul.SIMULATION_ID=$simId
            ");
    }

    $result = array();
    while($obj = $response->fetch_object()){
        $result[] = [
            "userName" => $obj->userName
        ];
    }

    echo json_encode($result);
    $db->close();
?>
