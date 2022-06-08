<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];

    if (!checkPw($db, $userName, $pw)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $userObj = $db->query("SELECT ID as id FROM user WHERE NAME ='".addslashes($userName)."'")->fetch_object();
    if (!$userObj) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $response = $db->query(
        "SELECT 
            SIMULATION_ID as id
        FROM
            userlike
        WHERE
            USER_ID={$userObj->id}
        ");

    $result = array();
    while($obj = $response->fetch_object()){
        $result[] = [
            "id" => (int)$obj->id
        ];
    }
    
    echo json_encode($result);
    $db->close();
?>
