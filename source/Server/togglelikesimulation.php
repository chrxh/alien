<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $simId = $_POST["simId"];

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

    $success = false;
    $likeResponse = $db->query("SELECT ID as id FROM userlike WHERE USER_ID = ".$userObj->id." AND SIMULATION_ID = " . addslashes($simId));
    $likeObj = $likeResponse->fetch_object();
    if ($likeObj) {
        if ($db->query("DELETE FROM userlike WHERE ID = $likeObj->id")) {
            $success = true;
        }
    }
    else {
        if ($db->query("INSERT INTO userlike (ID, USER_ID, SIMULATION_ID) VALUES (NULL, ".$userObj->id.", ".addslashes($simId).")")) {
            $success = true;
        }
    }
    echo json_encode(["result"=>$success]);

    $db->commit();
    $db->close();
?>