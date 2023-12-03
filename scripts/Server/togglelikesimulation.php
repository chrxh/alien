<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $simId = $_POST["simId"];
    $likeType = 0;
    if (array_key_exists("likeType", $_POST)) {
        $likeType = (int)$_POST["likeType"];
    }


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

    $success = true;
    $likeResponse = $db->query("SELECT ID as id, TYPE as likeType FROM userlike WHERE USER_ID = ".$userObj->id." AND SIMULATION_ID = " . addslashes($simId));
    $likeObj = $likeResponse->fetch_object();
    $onlyRemoveLike = false;
    if ($likeObj) {
        $origLikeType = is_null($likeObj->likeType) ? 0 : (int)$likeObj->likeType;
        if ($origLikeType == $likeType) {
            $onlyRemoveLike = true;
        }
        if (!$db->query("DELETE FROM userlike WHERE ID = $likeObj->id")) {
            $success = false;
        }
    }
    if (!$onlyRemoveLike) {
        if (!$db->query("INSERT INTO userlike (ID, USER_ID, SIMULATION_ID, TYPE) VALUES (NULL, ".$userObj->id.", ".addslashes($simId).", ".$likeType.")")) {
            $success = false;
        }
    }
    echo json_encode(["result"=>$success]);

    $db->commit();
    $db->close();
?>